# app.py - Version complète avec fallback intelligent
import os
import math
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

from flask import Flask, request, jsonify
from flask_cors import CORS
from supabase import create_client, Client
from pydantic import BaseModel
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# ==================== CONFIGURATION ====================

class Config:
    DEFAULT_LIMIT = 10
    MAX_LIMIT = 30
    MAX_CANDIDATES = 50

# ==================== MODÈLES ====================

class RecommendationRequest(BaseModel):
    user_id: Optional[str] = None
    limit: int = Config.DEFAULT_LIMIT
    city: Optional[str] = None
    neighborhood: Optional[str] = None
    budget_min: Optional[float] = None
    budget_max: Optional[float] = None
    property_type: Optional[str] = None

class FeedbackRequest(BaseModel):
    user_id: str
    property_id: str
    event_type: str  # 'view', 'favorite', 'contact'

# ==================== SCORING OPTIMISÉ ====================

class OptimizedScoringEngine:
    """Scoring amélioré : ville + quartier + type + budget + nouveautés + popularité"""
    
    def score_property(self, property: Dict, user_prefs: Dict) -> Dict:
        score = 0
        reasons = []

        # 1. Budget (40 points max)
        if user_prefs.get('budget_min') and user_prefs.get('budget_max'):
            price = property.get('price', 0)
            if user_prefs['budget_min'] <= price <= user_prefs['budget_max']:
                score += 40
                reasons.append("budget_match")
            elif price < user_prefs['budget_min']:
                score += 20
                reasons.append("good_deal")

        # 2. Ville + Quartier combinés (40 points max)
        city_match = user_prefs.get('city') and property.get('city', '').lower() == user_prefs['city'].lower()
        neighborhood_match = user_prefs.get('neighborhood') and property.get('neighborhood', '').lower() == user_prefs['neighborhood'].lower()

        if city_match and neighborhood_match:
            score += 40
            reasons.append("city+neighborhood_match")
        elif city_match:
            score += 25
            reasons.append("city_match")
        elif neighborhood_match:
            score += 25
            reasons.append("neighborhood_match")

        # 3. Type de propriété (20 points max)
        if user_prefs.get('property_type') and property.get('property_type') == user_prefs['property_type']:
            score += 20
            reasons.append("type_match")

        # 4. Boost nouveauté (10 points max)
        days_since_created = self._days_since(property.get('created_at'))
        if days_since_created < 3:
            score += 10
            reasons.append("new_listing")
        elif days_since_created < 7:
            score += 5
            reasons.append("recent")

        # 5. Popularité (bonus)
        if property.get('view_count', 0) > 50:
            score += 5
            reasons.append("trending")

        return {
            'score': score,
            'reasons': reasons[:3]
        }

    @staticmethod
    def _days_since(date_str: Optional[str]) -> float:
        if not date_str:
            return 30
        try:
            date = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
            return (datetime.now(timezone.utc) - date).total_seconds() / 86400
        except:
            return 30

# ==================== SERVICE DE RECOMMANDATION ====================

class RecommendationService:
    def __init__(self, supabase_client: Client):
        self.supabase = supabase_client
        self.scoring = OptimizedScoringEngine()
    
    def get_recommendations(self, req: RecommendationRequest) -> Dict:
        """
        Logique principale :
        1. Si user_id sans profil → fallback générique (récentes + populaires)
        2. Si profil avec critères → recherche personnalisée
        3. Si recherche vide → fallback similaire (même ville, budget ±20%)
        """
        
        # Déterminer si on a des préférences utilisateur
        has_preferences = any([
            req.city,
            req.neighborhood,
            req.budget_min,
            req.budget_max,
            req.property_type
        ])
        
        logger.info(f"Requête - user_id: {req.user_id}, has_preferences: {has_preferences}")
        
        # CAS 1 : Pas de préférences → fallback générique
        if not has_preferences:
            return self._get_generic_recommendations(req.limit)
        
        # CAS 2 : Préférences présentes → recherche personnalisée
        results = self._search_with_preferences(req)
        
        # CAS 3 : Aucun résultat → fallback similaire
        if not results['recommendations']:
            logger.info("Aucun résultat exact, fallback similaire...")
            return self._get_similar_fallback(req)
        
        return results
    
    def _search_with_preferences(self, req: RecommendationRequest) -> Dict:
        """Recherche avec critères exacts"""
        query = self.supabase.table('properties')\
            .select('*')\
            .eq('is_published', True)\
            .eq('is_available', True)
        
        # Filtres stricts
        if req.city:
            query = query.ilike('city', f'%{req.city}%')
        if req.neighborhood:
            query = query.ilike('neighborhood', f'%{req.neighborhood}%')
        if req.budget_min:
            query = query.gte('price', req.budget_min)
        if req.budget_max:
            query = query.lte('price', req.budget_max)
        if req.property_type:
            query = query.eq('property_type', req.property_type)
        
        response = query.order('created_at', desc=True).limit(Config.MAX_CANDIDATES).execute()
        candidates = response.data or []
        
        if not candidates:
            return {'recommendations': [], 'total': 0, 'is_fallback': False}
        
        # Scoring
        user_prefs = {
            'city': req.city,
            'neighborhood': req.neighborhood,
            'budget_min': req.budget_min,
            'budget_max': req.budget_max,
            'property_type': req.property_type
        }
        
        scored = []
        for prop in candidates:
            scoring = self.scoring.score_property(prop, user_prefs)
            scored.append({
                **prop,
                '_score': scoring['score'],
                '_reasons': scoring['reasons']
            })
        
        scored.sort(key=lambda x: x['_score'], reverse=True)
        limit = min(req.limit, Config.MAX_LIMIT)
        
        return {
            'recommendations': scored[:limit],
            'total': len(scored),
            'is_fallback': False
        }
    
    def _get_similar_fallback(self, req: RecommendationRequest) -> Dict:
        """Fallback : même ville, budget ±20%, sans quartier"""
        query = self.supabase.table('properties')\
            .select('*')\
            .eq('is_published', True)\
            .eq('is_available', True)
        
        # Critères élargis
        if req.city:
            query = query.ilike('city', f'%{req.city}%')
        
        # Budget élargi ±20%
        if req.budget_min:
            query = query.gte('price', req.budget_min * 0.8)
        if req.budget_max:
            query = query.lte('price', req.budget_max * 1.2)
        
        # Type conservé si possible
        if req.property_type:
            query = query.eq('property_type', req.property_type)
        
        response = query.order('created_at', desc=True).limit(Config.MAX_CANDIDATES).execute()
        candidates = response.data or []
        
        if not candidates:
            # Dernier recours : générique
            return self._get_generic_recommendations(req.limit)
        
        # Scoring avec critères élargis
        user_prefs = {
            'city': req.city,
            'budget_min': req.budget_min,
            'budget_max': req.budget_max,
            'property_type': req.property_type
        }
        
        scored = []
        for prop in candidates:
            scoring = self.scoring.score_property(prop, user_prefs)
            scored.append({
                **prop,
                '_score': scoring['score'],
                '_reasons': scoring['reasons'],
                '_is_similar_fallback': True
            })
        
        scored.sort(key=lambda x: x['_score'], reverse=True)
        limit = min(req.limit, Config.MAX_LIMIT)
        
        return {
            'recommendations': scored[:limit],
            'total': len(scored),
            'is_fallback': True,
            'fallback_type': 'similar',
            'message': 'Aucune propriété exacte trouvée. Voici des suggestions proches.'
        }
    
    def _get_generic_recommendations(self, limit: int) -> Dict:
        """Fallback générique : récentes + populaires"""
        logger.info("Fallback générique - récentes et populaires")
        
        # Récupérer les plus récentes et les plus vues
        response = self.supabase.table('properties')\
            .select('*')\
            .eq('is_published', True)\
            .eq('is_available', True)\
            .order('created_at', desc=True)\
            .limit(Config.MAX_CANDIDATES)\
            .execute()
        
        candidates = response.data or []
        
        if not candidates:
            return {
                'recommendations': [],
                'total': 0,
                'is_fallback': True,
                'fallback_type': 'generic'
            }
        
        # Scoring basé sur nouveauté et popularité uniquement
        scored = []
        for prop in candidates:
            score = 0
            reasons = []
            
            # Nouveauté
            days = self.scoring._days_since(prop.get('created_at'))
            if days < 3:
                score += 50
                reasons.append("new_listing")
            elif days < 7:
                score += 30
                reasons.append("recent")
            
            # Popularité
            views = prop.get('view_count', 0)
            if views > 100:
                score += 30
                reasons.append("very_popular")
            elif views > 50:
                score += 20
                reasons.append("popular")
            elif views > 20:
                score += 10
                reasons.append("trending")
            
            scored.append({
                **prop,
                '_score': score,
                '_reasons': reasons[:3],
                '_is_generic_fallback': True
            })
        
        scored.sort(key=lambda x: x['_score'], reverse=True)
        final_limit = min(limit, Config.MAX_LIMIT)
        
        return {
            'recommendations': scored[:final_limit],
            'total': len(scored),
            'is_fallback': True,
            'fallback_type': 'generic',
            'message': 'Bienvenue ! Voici les propriétés les plus récentes et populaires.'
        }

# ==================== INITIALISATION ====================

supabase_url = os.getenv('SUPABASE_URL')
supabase_key = os.getenv('SUPABASE_SERVICE_ROLE_KEY')

if not supabase_url or not supabase_key:
    logger.warning("Variables d'environnement Supabase manquantes")
    supabase = None
    recommendation_service = None
else:
    supabase: Client = create_client(supabase_url, supabase_key)
    recommendation_service = RecommendationService(supabase)

# ==================== ROUTES ====================

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'ok',
        'supabase': 'connected' if supabase else 'disconnected',
        'service': 'ready' if recommendation_service else 'unavailable'
    })

@app.route('/recommendations', methods=['POST'])
def get_recommendations():
    if not recommendation_service:
        return jsonify({'error': 'Service indisponible'}), 503
    
    try:
        body = request.get_json() or {}
        req = RecommendationRequest(**body)
        
        # Utiliser le service de recommandation
        result = recommendation_service.get_recommendations(req)
        
        # Ajouter les métadonnées de réponse
        response = {
            'recommendations': result.get('recommendations', []),
            'total': result.get('total', 0),
            'is_fallback': result.get('is_fallback', False),
        }
        
        # Ajouter les champs optionnels
        if result.get('fallback_type'):
            response['fallback_type'] = result['fallback_type']
        if result.get('message'):
            response['message'] = result['message']
        
        return jsonify(response)
        
    except Exception as e:
        logger.exception("Erreur dans /recommendations")
        return jsonify({'error': str(e)}), 500

@app.route('/feedback', methods=['POST'])
def track_feedback():
    if not supabase:
        return jsonify({'error': 'Service indisponible'}), 503
    
    try:
        body = request.get_json() or {}
        feedback = FeedbackRequest(**body)
        
        supabase.table('feedback_events').insert({
            'user_id': feedback.user_id,
            'property_id': feedback.property_id,
            'event_type': feedback.event_type,
            'created_at': datetime.now(timezone.utc).isoformat()
        }).execute()
        
        return jsonify({'success': True})
        
    except Exception as e:
        logger.exception("Feedback error")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.getenv('PORT', 10000))
    app.run(host='0.0.0.0', port=port)

# app.py - Version starter pour Habynex
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
    budget_min: Optional[float] = None
    budget_max: Optional[float] = None
    property_type: Optional[str] = None  # 'apartment', 'house', etc.

class FeedbackRequest(BaseModel):
    user_id: str
    property_id: str
    event_type: str  # 'view', 'favorite', 'contact'

# ==================== SCORING SIMPLE ====================

class SimpleScoringEngine:
    """Scoring basique : 3 critères principaux"""
    
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
                score += 20  # Bonne affaire
                reasons.append("good_deal")
        
        # 2. Localisation (30 points max)
        if user_prefs.get('city'):
            if property.get('city', '').lower() == user_prefs['city'].lower():
                score += 30
                reasons.append("city_match")
        
        # 3. Type de propriété (20 points max)
        if user_prefs.get('property_type'):
            if property.get('property_type') == user_prefs['property_type']:
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
        view_count = property.get('view_count', 0)
        if view_count > 50:
            score += 5
            reasons.append("trending")
        
        return {
            'score': score,
            'reasons': reasons[:3]  # Top 3 raisons
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

# ==================== ROUTES ====================

# Initialisation Supabase
supabase_url = os.getenv('SUPABASE_URL')
supabase_key = os.getenv('SUPABASE_SERVICE_ROLE_KEY')

if not supabase_url or not supabase_key:
    logger.warning("Variables d'environnement Supabase manquantes")
    supabase = None
else:
    supabase: Client = create_client(supabase_url, supabase_key)

scoring_engine = SimpleScoringEngine()

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'ok',
        'supabase': 'connected' if supabase else 'disconnected'
    })

@app.route('/recommendations', methods=['POST'])
def get_recommendations():
    if not supabase:
        return jsonify({'error': 'Service indisponible'}), 503
    try:
        body = request.get_json() or {}
        req = RecommendationRequest(**body)
        
        # 1. CONSTRUIRE LA REQUÊTE DE BASE
        query = supabase.table('properties')\
            .select('*')\
            .eq('is_published', True)\
            .eq('is_available', True)
        
        # 2. FILTRES DE BASE
        if req.city:
            query = query.ilike('city', f'%{req.city}%')
        
        if req.budget_min:
            query = query.gte('price', req.budget_min * 0.8)
        if req.budget_max:
            query = query.lte('price', req.budget_max * 1.2)
        
        if req.property_type:
            query = query.eq('property_type', req.property_type)
        
        # 3. RÉCUPÉRER LES CANDIDATS
        response = query.order('created_at', desc=True).limit(Config.MAX_CANDIDATES).execute()
        candidates = response.data or []
        
        if not candidates:
            return jsonify({
                'recommendations': [],
                'total': 0,
                'message': 'Aucune propriété trouvée'
            })
        
        # 4. SCORER ET TRIER
        user_prefs = {
            'city': req.city,
            'budget_min': req.budget_min,
            'budget_max': req.budget_max,
            'property_type': req.property_type
        }
        
        scored = []
        for prop in candidates:
            scoring = scoring_engine.score_property(prop, user_prefs)
            scored.append({
                **prop,
                '_score': scoring['score'],
                '_reasons': scoring['reasons']
            })
        
        # Trier par score décroissant
        scored.sort(key=lambda x: x['_score'], reverse=True)
        
        # 5. PAGINATION
        limit = min(req.limit, Config.MAX_LIMIT)
        results = scored[:limit]
        
        return jsonify({
            'recommendations': results,
            'total': len(scored),
            'filters_applied': {
                'city': req.city,
                'budget_range': [req.budget_min, req.budget_max] if req.budget_min or req.budget_max else None,
                'property_type': req.property_type
            }
        })
        
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
        
        # Sauvegarder l'événement pour analyse future
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

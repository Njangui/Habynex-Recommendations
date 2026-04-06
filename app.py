# app.py - Version corrigée et fonctionnelle (alignée avec le frontend)
import os
import math
import json
import hashlib
import time
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from functools import lru_cache

from flask import Flask, request, jsonify
from flask_cors import CORS
from supabase import create_client, Client
from pydantic import BaseModel
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# ==================== CACHE SIMPLE EN MÉMOIRE ====================
class SimpleCache:
    def __init__(self, ttl_seconds: int = 300):
        self._cache = {}
        self._ttl = ttl_seconds
    
    def _make_key(self, *args, **kwargs) -> str:
        key_data = json.dumps({"args": args, "kwargs": kwargs}, sort_keys=True)
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def get(self, *args, **kwargs) -> Optional[Any]:
        key = self._make_key(*args, **kwargs)
        if key in self._cache:
            data, timestamp = self._cache[key]
            if time.time() - timestamp < self._ttl:
                logger.info(f"Cache HIT: {key[:8]}")
                return data
            else:
                del self._cache[key]
        return None
    
    def set(self, value: Any, *args, **kwargs):
        key = self._make_key(*args, **kwargs)
        self._cache[key] = (value, time.time())
        logger.info(f"Cache SET: {key[:8]}")

    def invalidate(self, pattern: str = None):
        if pattern:
            keys_to_delete = [k for k in self._cache.keys() if pattern in k]
            for k in keys_to_delete:
                del self._cache[k]
        else:
            self._cache.clear()

recommendations_cache = SimpleCache(ttl_seconds=300)
profile_cache = SimpleCache(ttl_seconds=60)

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
    listing_type: Optional[str] = None

class FeedbackRequest(BaseModel):
    user_id: str
    property_id: str
    event_type: str

# ==================== SERVICE PROFIL ====================
class ProfileService:
    def __init__(self, supabase_client: Client):
        self.supabase = supabase_client
    
    def get_profile(self, user_id: str) -> Optional[Dict]:
        if not user_id:
            return None
        
        cached = profile_cache.get("profile", user_id)
        if cached:
            return cached
        
        try:
            # ✅ CORRECTION : Utiliser 'user_id' comme colonne (aligné avec le frontend)
            # et les mêmes noms de colonnes que le frontend
            response = self.supabase.table("profiles")\
                .select("""
                    user_id,
                    city, 
                    preferred_neighborhoods, 
                    budget_min, 
                    budget_max, 
                    preferred_property_types,
                    preferred_listing_types
                """)\
                .eq("user_id", user_id)\
                .limit(1)\
                .execute()
            
            # ✅ CORRECTION : Gestion robuste de la réponse
            profile = None
            if response and hasattr(response, 'data') and response.data:
                profile = response.data[0] if len(response.data) > 0 else None
            
            if profile:
                profile_cache.set(profile, "profile", user_id)
                logger.info(f"✅ Profil récupéré: {profile.get('city')} pour user_id={user_id}")
            else:
                logger.info(f"❌ Profil non trouvé pour user_id: {user_id}")
            
            return profile
            
        except Exception as e:
            logger.error(f"💥 Erreur récupération profil {user_id}: {e}")
            return None

# ==================== SCORING OPTIMISÉ ====================
class OptimizedScoringEngine:
    WEIGHTS = {
        'budget': 35,
        'location': 35,
        'property_type': 20,
        'recency': 10,
        'popularity': 5
    }
    
    def score_property(self, property: Dict, user_prefs: Dict, is_fallback: bool = False) -> Dict:
        score = 0
        reasons = []
        
        # 1. Budget (35 points max)
        if user_prefs.get('budget_min') is not None and user_prefs.get('budget_max') is not None:
            price = property.get('price', 0)
            budget_min = user_prefs['budget_min']
            budget_max = user_prefs['budget_max']
            
            if budget_min <= price <= budget_max:
                score += self.WEIGHTS['budget']
                reasons.append("budget_match")
            elif price < budget_min * 0.9:
                score += self.WEIGHTS['budget'] * 0.7
                reasons.append("good_deal")
            elif price <= budget_max * 1.1:
                score += self.WEIGHTS['budget'] * 0.4
                reasons.append("slightly_over_budget")
        
        # 2. Localisation (35 points max)
        prop_city = (property.get('city') or '').lower().strip()
        prop_neighborhood = (property.get('neighborhood') or '').lower().strip()
        user_city = (user_prefs.get('city') or '').lower().strip()
        user_neighborhood = (user_prefs.get('neighborhood') or '').lower().strip()
        
        city_match = user_city and prop_city == user_city
        neighborhood_match = user_neighborhood and prop_neighborhood == user_neighborhood
        
        if city_match and neighborhood_match:
            score += self.WEIGHTS['location']
            reasons.append("city+neighborhood_match")
        elif city_match:
            score += self.WEIGHTS['location'] * 0.7
            reasons.append("city_match")
        elif neighborhood_match:
            score += self.WEIGHTS['location'] * 0.5
            reasons.append("neighborhood_match")
        
        # 3. Type de propriété (20 points max)
        if user_prefs.get('property_type') and property.get('property_type') == user_prefs['property_type']:
            score += self.WEIGHTS['property_type']
            reasons.append("type_match")
        
        # 4. Nouveauté (10 points max)
        days_since = self._days_since(property.get('created_at'))
        if days_since < 3:
            score += self.WEIGHTS['recency']
            reasons.append("new_listing")
        elif days_since < 7:
            score += self.WEIGHTS['recency'] * 0.5
            reasons.append("recent")
        
        # 5. Popularité (5 points max)
        view_count = property.get('view_count', 0)
        if view_count > 100:
            score += self.WEIGHTS['popularity']
            reasons.append("very_popular")
        elif view_count > 50:
            score += self.WEIGHTS['popularity'] * 0.6
            reasons.append("popular")
        
        # Bonus correspondance parfaite
        if len(reasons) >= 4:
            score += 10
            reasons.append("perfect_match")
        
        return {
            'score': min(score, 100),
            'reasons': reasons[:3],
            'is_fallback': is_fallback
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
        self.profile_service = ProfileService(supabase_client)
    
    def get_recommendations(self, req: RecommendationRequest) -> Dict:
        logger.info(f"=== NOUVELLE REQUÊTE ===")
        logger.info(f"user_id: {req.user_id}")
        logger.info(f"Reçu: city={req.city}, budget={req.budget_min}-{req.budget_max}, type={req.property_type}")
        
        # Récupérer le profil si user_id fourni
        user_profile = None
        if req.user_id:
            user_profile = self.profile_service.get_profile(req.user_id)
            logger.info(f"Profil récupéré: {user_profile is not None}")
        
        # Fusionner les préférences (requête > profil)
        merged_prefs = self._merge_preferences(req, user_profile)
        logger.info(f"Préférences fusionnées: {merged_prefs}")
        
        # Vérifier si on a des préférences significatives
        has_preferences = self._has_significant_preferences(merged_prefs)
        logger.info(f"has_preferences: {has_preferences}")
        
        # Clé de cache
        cache_key = (
            req.user_id or 'anonymous',
            merged_prefs.get('city'),
            merged_prefs.get('neighborhood'),
            merged_prefs.get('budget_min'),
            merged_prefs.get('budget_max'),
            merged_prefs.get('property_type'),
            req.limit
        )
        
        # Vérifier cache
        cached_result = recommendations_cache.get(*cache_key)
        if cached_result:
            cached_result['from_cache'] = True
            logger.info("Retour cache")
            return cached_result
        
        # PAS DE PRÉFÉRENCES → fallback générique
        if not has_preferences:
            logger.info("🔴 PAS DE PRÉFÉRENCES → Fallback générique")
            result = self._get_generic_recommendations(req.limit)
            recommendations_cache.set(result, *cache_key)
            return result
        
        # RECHERCHE PERSONNALISÉE
        logger.info("🟢 PRÉFÉRENCES TROUVÉES → Recherche personnalisée")
        results = self._search_with_preferences(merged_prefs, req.limit)
        logger.info(f"Résultats recherche: {len(results['recommendations'])}")
        
        # Si peu de résultats, fallback amélioré
        if len(results['recommendations']) < 3:
            logger.info("🟡 PEU DE RÉSULTATS → Fallback amélioré")
            fallback_results = self._get_similar_fallback(merged_prefs, req.limit)
            
            # Fusionner sans doublons
            seen_ids = {r['id'] for r in results['recommendations']}
            for r in fallback_results['recommendations']:
                if r['id'] not in seen_ids:
                    results['recommendations'].append(r)
                    seen_ids.add(r['id'])
            
            results['recommendations'] = results['recommendations'][:req.limit]
            results['is_fallback'] = True
            results['fallback_type'] = 'enhanced'
            results['message'] = 'Voici les meilleures correspondances, y compris des suggestions proches.'
        
        recommendations_cache.set(results, *cache_key)
        return results
    
    def _has_significant_preferences(self, prefs: Dict) -> bool:
        """Détecter si on a des préférences significatives"""
        has_city = bool(prefs.get('city'))
        has_neighborhood = bool(prefs.get('neighborhood'))
        has_budget = prefs.get('budget_min') is not None or prefs.get('budget_max') is not None
        has_property_type = bool(prefs.get('property_type'))
        
        # On considère qu'on a des préférences si on a AU MOINS ville OU budget
        return has_city or has_budget or has_neighborhood or has_property_type
    
    def _merge_preferences(self, req: RecommendationRequest, profile: Optional[Dict]) -> Dict:
        merged = {}
        
        # Extraire du profil (utilise les mêmes noms que le frontend)
        if profile:
            merged['city'] = profile.get('city')
            
            # preferred_neighborhoods est un tableau (comme dans le frontend)
            neighborhoods = profile.get('preferred_neighborhoods', [])
            if neighborhoods and len(neighborhoods) > 0:
                merged['neighborhood'] = neighborhoods[0]
            
            merged['budget_min'] = profile.get('budget_min')
            merged['budget_max'] = profile.get('budget_max')
            
            # preferred_property_types est un tableau (comme dans le frontend)
            property_types = profile.get('preferred_property_types', [])
            if property_types and len(property_types) > 0:
                merged['property_type'] = property_types[0]
            
            # preferred_listing_types est un tableau (comme dans le frontend)
            listing_types = profile.get('preferred_listing_types', [])
            if listing_types and len(listing_types) > 0:
                merged['listing_type'] = listing_types[0]
        
        # Requête écrase le profil
        if req.city is not None:
            merged['city'] = req.city
        if req.neighborhood is not None:
            merged['neighborhood'] = req.neighborhood
        if req.budget_min is not None:
            merged['budget_min'] = req.budget_min
        if req.budget_max is not None:
            merged['budget_max'] = req.budget_max
        if req.property_type is not None:
            merged['property_type'] = req.property_type
        if req.listing_type is not None:
            merged['listing_type'] = req.listing_type
        
        return merged
    
    def _search_with_preferences(self, prefs: Dict, limit: int) -> Dict:
        """Recherche avec critères exacts"""
        query = self.supabase.table('properties')\
            .select('*')\
            .eq('is_published', True)\
            .eq('is_available', True)
        
        # Filtres flexibles
        if prefs.get('city'):
            query = query.ilike('city', f"%{prefs['city']}%")
        if prefs.get('neighborhood'):
            query = query.ilike('neighborhood', f"%{prefs['neighborhood']}%")
        if prefs.get('budget_min') is not None:
            query = query.gte('price', prefs['budget_min'])
        if prefs.get('budget_max') is not None:
            query = query.lte('price', prefs['budget_max'])
        if prefs.get('property_type'):
            query = query.eq('property_type', prefs['property_type'])
        if prefs.get('listing_type'):
            query = query.eq('listing_type', prefs['listing_type'])
        
        response = query.order('created_at', desc=True).limit(Config.MAX_CANDIDATES).execute()
        candidates = response.data or []
        
        logger.info(f"Candidats trouvés: {len(candidates)}")
        
        if not candidates:
            return {'recommendations': [], 'total': 0, 'is_fallback': False}
        
        # Scoring
        scored = []
        for prop in candidates:
            scoring = self.scoring.score_property(prop, prefs, is_fallback=False)
            if scoring['score'] > 20:  # Seulement les scores > 20
                scored.append({
                    **prop,
                    '_score': scoring['score'],
                    '_reasons': scoring['reasons'],
                    '_is_fallback': False
                })
        
        scored.sort(key=lambda x: x['_score'], reverse=True)
        final_limit = min(limit, Config.MAX_LIMIT)
        
        return {
            'recommendations': scored[:final_limit],
            'total': len(scored),
            'is_fallback': False
        }
    
    def _get_similar_fallback(self, prefs: Dict, limit: int) -> Dict:
        """Fallback élargi mais intelligent"""
        query = self.supabase.table('properties')\
            .select('*')\
            .eq('is_published', True)\
            .eq('is_available', True)
        
        # Critères élargis
        if prefs.get('city'):
            query = query.ilike('city', f"%{prefs['city']}%")
        
        # Budget élargi ±30%
        if prefs.get('budget_min') is not None:
            query = query.gte('price', prefs['budget_min'] * 0.7)
        if prefs.get('budget_max') is not None:
            query = query.lte('price', prefs['budget_max'] * 1.3)
        
        if prefs.get('property_type'):
            query = query.eq('property_type', prefs['property_type'])
        
        response = query.order('created_at', desc=True).limit(Config.MAX_CANDIDATES).execute()
        candidates = response.data or []
        
        logger.info(f"Fallback candidats: {len(candidates)}")
        
        if not candidates:
            return self._get_generic_recommendations(limit)
        
        # Scoring avec critères élargis
        scored = []
        for prop in candidates:
            scoring = self.scoring.score_property(prop, prefs, is_fallback=True)
            scored.append({
                **prop,
                '_score': scoring['score'] * 0.8,  # Pénalité 20%
                '_reasons': scoring['reasons'],
                '_is_fallback': True
            })
        
        scored.sort(key=lambda x: x['_score'], reverse=True)
        final_limit = min(limit, Config.MAX_LIMIT)
        
        return {
            'recommendations': scored[:final_limit],
            'total': len(scored),
            'is_fallback': True,
            'fallback_type': 'similar'
        }
    
    def _get_generic_recommendations(self, limit: int) -> Dict:
        """Fallback générique: récentes + populaires"""
        logger.info("Fallback générique")
        
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
        
        scored = []
        for prop in candidates:
            score = 0
            reasons = []
            
            days = self.scoring._days_since(prop.get('created_at'))
            if days < 3:
                score += 50
                reasons.append("new_listing")
            elif days < 7:
                score += 30
                reasons.append("recent")
            
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
                '_is_fallback': True,
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
        
        logger.info(f"Requête reçue: {req.dict()}")
        
        result = recommendation_service.get_recommendations(req)
        
        response = {
            'recommendations': result.get('recommendations', []),
            'total': result.get('total', 0),
            'is_fallback': result.get('is_fallback', False),
            'from_cache': result.get('from_cache', False)
        }
        
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
        
        recommendations_cache.invalidate(feedback.user_id)
        
        return jsonify({'success': True})
        
    except Exception as e:
        logger.exception("Feedback error")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.getenv('PORT', 10000))
    app.run(host='0.0.0.0', port=port)

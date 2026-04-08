import os
import math
import json
import hashlib
import time
import unicodedata
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from functools import lru_cache

from flask import Flask, request, jsonify
from flask_cors import CORS
from supabase import create_client, Client
from pydantic import BaseModel
import logging
import threading

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# ==================== CONFIGURATION OPTIMISÉE ====================
class Config:
    DEFAULT_LIMIT = 10
    MAX_LIMIT = 20  # Réduit de 30 à 20
    MAX_CANDIDATES = 20  # Réduit de 50 à 20
    REQUEST_TIMEOUT = 8  # Timeout interne en secondes (Render a ~10s)
    CACHE_TTL = 600  # Cache plus long (10 min) pour réduire les appels

# ==================== CACHE AMÉLIORÉ ====================
class SimpleCache:
    def __init__(self, ttl_seconds: int = Config.CACHE_TTL):
        self._cache = {}
        self._ttl = ttl_seconds
        self._lock = threading.Lock()  # Thread-safe
    
    def _make_key(self, *args, **kwargs) -> str:
        key_data = json.dumps({"args": args, "kwargs": kwargs}, sort_keys=True)
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def get(self, *args, **kwargs) -> Optional[Any]:
        key = self._make_key(*args, **kwargs)
        with self._lock:
            if key in self._cache:
                data, timestamp = self._cache[key]
                if time.time() - timestamp < self._ttl:
                    logger.info(f"✅ Cache HIT: {key[:8]}")
                    return data
                else:
                    del self._cache[key]
        return None
    
    def set(self, value: Any, *args, **kwargs):
        key = self._make_key(*args, **kwargs)
        with self._lock:
            self._cache[key] = (value, time.time())
        logger.info(f"💾 Cache SET: {key[:8]}")

    def invalidate(self, pattern: str = None):
        with self._lock:
            if pattern:
                keys_to_delete = [k for k in self._cache.keys() if pattern in k]
                for k in keys_to_delete:
                    del self._cache[k]
            else:
                self._cache.clear()

recommendations_cache = SimpleCache()
profile_cache = SimpleCache(ttl_seconds=300)

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

# ==================== UTILITAIRES ====================
def normalize_text(text: Optional[str]) -> str:
    """Normalise le texte : minuscules, sans accents"""
    if not text:
        return ""
    text = text.lower()
    text = ''.join(c for c in unicodedata.normalize('NFD', text) if unicodedata.category(c) != 'Mn')
    return text.strip()

# ==================== SERVICE PROFIL OPTIMISÉ ====================
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
            # 🚀 OPTIMISATION : Requête plus légère
            response = self.supabase.table("profiles")\
                .select("user_id, city, preferred_neighborhoods, budget_min, budget_max, preferred_property_types, preferred_listing_types")\
                .eq("user_id", user_id)\
                .limit(1)\
                .execute()
            
            profile = response.data[0] if response.data else None
            
            if profile:
                profile_cache.set(profile, "profile", user_id)
                logger.info(f"✅ Profil: {profile.get('city')} pour {user_id[:8]}...")
            
            return profile
            
        except Exception as e:
            logger.error(f"❌ Erreur profil: {e}")
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
        
        # 1. Budget (35 points)
        price = property.get('price', 0)
        budget_min = user_prefs.get('budget_min')
        budget_max = user_prefs.get('budget_max')
        
        if budget_min is not None and budget_max is not None:
            if budget_min <= price <= budget_max:
                score += self.WEIGHTS['budget']
                reasons.append("budget_match")
            elif price < budget_min * 0.9:
                score += self.WEIGHTS['budget'] * 0.7
                reasons.append("good_deal")
            elif price <= budget_max * 1.1:
                score += self.WEIGHTS['budget'] * 0.4
                reasons.append("slightly_over")
        
        # 2. Localisation (35 points) - NORMALISATION
        prop_city = normalize_text(property.get('city', ''))
        prop_neighborhood = normalize_text(property.get('neighborhood', ''))
        user_city = normalize_text(user_prefs.get('city', ''))
        user_neighborhood = normalize_text(user_prefs.get('neighborhood', ''))
        
        if user_city and prop_city == user_city:
            score += self.WEIGHTS['location'] * 0.7
            reasons.append("city_match")
            if user_neighborhood and prop_neighborhood == user_neighborhood:
                score += self.WEIGHTS['location'] * 0.3
                reasons.append("neighborhood_match")
        
        # 3. Type (20 points)
        if user_prefs.get('property_type') and property.get('property_type') == user_prefs['property_type']:
            score += self.WEIGHTS['property_type']
            reasons.append("type_match")
        
        # 4. Recency (10 points)
        days = self._days_since(property.get('created_at'))
        if days < 3:
            score += self.WEIGHTS['recency']
            reasons.append("new")
        elif days < 7:
            score += self.WEIGHTS['recency'] * 0.5
            reasons.append("recent")
        
        # 5. Popularité (5 points)
        views = property.get('view_count', 0)
        if views > 50:
            score += self.WEIGHTS['popularity']
            reasons.append("popular")
        
        return {
            'score': min(score, 100),
            'reasons': reasons[:2],  # Limite à 2 raisons
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

# ==================== SERVICE RECOMMANDATION OPTIMISÉ ====================
class RecommendationService:
    def __init__(self, supabase_client: Client):
        self.supabase = supabase_client
        self.scoring = OptimizedScoringEngine()
        self.profile_service = ProfileService(supabase_client)
    
    def get_recommendations(self, req: RecommendationRequest) -> Dict:
        start_time = time.time()
        logger.info(f"🚀 Requête: user={req.user_id[:8] if req.user_id else 'anon'}...")
        
        # 1. Vérifier cache immédiatement
        cache_key = self._make_cache_key(req)
        cached = recommendations_cache.get(*cache_key)
        if cached:
            logger.info(f"⚡ Cache hit en {time.time()-start_time:.2f}s")
            cached['from_cache'] = True
            return cached
        
        # 2. Récupérer profil (avec timeout implicite)
        user_profile = None
        if req.user_id:
            user_profile = self.profile_service.get_profile(req.user_id)
        
        # 3. Fusionner préférences
        prefs = self._merge_preferences(req, user_profile)
        
        # 4. Vérifier timeout
        if time.time() - start_time > Config.REQUEST_TIMEOUT:
            logger.warning("⏱️ Timeout imminent, fallback rapide")
            return self._fast_fallback(req.limit)
        
        # 5. Exécuter recherche
        has_prefs = self._has_significant_preferences(prefs)
        
        if not has_prefs:
            result = self._get_generic_recommendations(req.limit)
        else:
            result = self._search_with_preferences(prefs, req.limit)
            
            # Si peu de résultats, fallback rapide
            if len(result['recommendations']) < 3:
                fallback = self._get_similar_fallback(prefs, req.limit)
                # Fusion sans doublons
                seen = {r['id'] for r in result['recommendations']}
                for r in fallback['recommendations']:
                    if r['id'] not in seen and len(result['recommendations']) < req.limit:
                        result['recommendations'].append(r)
                        seen.add(r['id'])
                result['is_fallback'] = True
        
        # 6. Mettre en cache et retourner
        recommendations_cache.set(result, *cache_key)
        logger.info(f"✅ Terminé en {time.time()-start_time:.2f}s - {len(result['recommendations'])} résultats")
        
        return result
    
    def _make_cache_key(self, req: RecommendationRequest) -> tuple:
        return (
            req.user_id or 'anon',
            req.city,
            req.neighborhood,
            req.budget_min,
            req.budget_max,
            req.property_type,
            min(req.limit, Config.MAX_LIMIT)
        )
    
    def _has_significant_preferences(self, prefs: Dict) -> bool:
        return bool(
            prefs.get('city') or 
            prefs.get('neighborhood') or
            prefs.get('budget_min') is not None or
            prefs.get('budget_max') is not None or
            prefs.get('property_type')
        )
    
    def _merge_preferences(self, req: RecommendationRequest, profile: Optional[Dict]) -> Dict:
        merged = {}
        
        # Profil
        if profile:
            merged['city'] = profile.get('city')
            neighborhoods = profile.get('preferred_neighborhoods', [])
            if neighborhoods:
                merged['neighborhood'] = neighborhoods[0]
            merged['budget_min'] = profile.get('budget_min')
            merged['budget_max'] = profile.get('budget_max')
            types = profile.get('preferred_property_types', [])
            if types:
                merged['property_type'] = types[0]
        
        # Requête écrase profil
        if req.city: merged['city'] = req.city
        if req.neighborhood: merged['neighborhood'] = req.neighborhood
        if req.budget_min is not None: merged['budget_min'] = req.budget_min
        if req.budget_max is not None: merged['budget_max'] = req.budget_max
        if req.property_type: merged['property_type'] = req.property_type
        
        return merged
    
    def _search_with_preferences(self, prefs: Dict, limit: int) -> Dict:
        """Recherche optimisée avec index"""
        try:
            query = self.supabase.table('properties')\
                .select('id, title, price, city, neighborhood, property_type, created_at, view_count, images, bedrooms, bathrooms, surface')\
                .eq('is_published', True)\
                .eq('is_available', True)
            
            # 🚀 OPTIMISATION : Filtres exacts d'abord (utilisent les index)
            if prefs.get('property_type'):
                query = query.eq('property_type', prefs['property_type'])
            
            if prefs.get('budget_max'):
                query = query.lte('price', prefs['budget_max'])
            
            if prefs.get('budget_min'):
                query = query.gte('price', prefs['budget_min'])
            
            # 🚀 OPTIMISATION : Ordre par date (indexé) puis limite stricte
            response = query.order('created_at', desc=True).limit(Config.MAX_CANDIDATES).execute()
            candidates = response.data or []
            
            # Filtrage post-requête pour la ville (normalisation)
            if prefs.get('city'):
                city_norm = normalize_text(prefs['city'])
                candidates = [
                    p for p in candidates 
                    if city_norm in normalize_text(p.get('city', ''))
                ]
            
            # Scoring rapide
            scored = []
            for prop in candidates:
                scoring = self.scoring.score_property(prop, prefs)
                if scoring['score'] > 15:  # Seuil plus bas
                    scored.append({
                        **prop,
                        '_score': scoring['score'],
                        '_reasons': scoring['reasons']
                    })
            
            scored.sort(key=lambda x: x['_score'], reverse=True)
            
            return {
                'recommendations': scored[:limit],
                'total': len(scored),
                'is_fallback': False
            }
            
        except Exception as e:
            logger.error(f"❌ Erreur recherche: {e}")
            return {'recommendations': [], 'total': 0, 'is_fallback': False}
    
    def _get_similar_fallback(self, prefs: Dict, limit: int) -> Dict:
        """Fallback léger"""
        try:
            query = self.supabase.table('properties')\
                .select('id, title, price, city, neighborhood, property_type, created_at, view_count, images')\
                .eq('is_published', True)\
                .eq('is_available', True)
            
            # Budget élargi
            if prefs.get('budget_max'):
                query = query.lte('price', prefs['budget_max'] * 1.3)
            
            response = query.order('created_at', desc=True).limit(Config.MAX_CANDIDATES).execute()
            candidates = response.data or []
            
            # Filtrage ville
            if prefs.get('city'):
                city_norm = normalize_text(prefs['city'])
                candidates = [p for p in candidates if city_norm in normalize_text(p.get('city', ''))]
            
            scored = []
            for prop in candidates:
                scoring = self.scoring.score_property(prop, prefs, is_fallback=True)
                scored.append({
                    **prop,
                    '_score': scoring['score'] * 0.8,
                    '_reasons': scoring['reasons'],
                    '_is_fallback': True
                })
            
            scored.sort(key=lambda x: x['_score'], reverse=True)
            
            return {
                'recommendations': scored[:limit],
                'total': len(scored),
                'is_fallback': True,
                'fallback_type': 'similar'
            }
            
        except Exception as e:
            logger.error(f"❌ Erreur fallback: {e}")
            return self._get_generic_recommendations(limit)
    
    def _get_generic_recommendations(self, limit: int) -> Dict:
        """Fallback générique ultra-rapide"""
        try:
            response = self.supabase.table('properties')\
                .select('id, title, price, city, neighborhood, property_type, created_at, view_count, images')\
                .eq('is_published', True)\
                .eq('is_available', True)\
                .order('created_at', desc=True)\
                .limit(limit)\
                .execute()
            
            candidates = response.data or []
            
            # Scoring minimal
            scored = []
            for prop in candidates:
                score = 0
                reasons = []
                
                days = self.scoring._days_since(prop.get('created_at'))
                if days < 7:
                    score += 40
                    reasons.append("new")
                
                if prop.get('view_count', 0) > 30:
                    score += 20
                    reasons.append("popular")
                
                scored.append({
                    **prop,
                    '_score': score,
                    '_reasons': reasons,
                    '_is_fallback': True,
                    '_is_generic_fallback': True
                })
            
            scored.sort(key=lambda x: x['_score'], reverse=True)
            
            return {
                'recommendations': scored[:limit],
                'total': len(scored),
                'is_fallback': True,
                'fallback_type': 'generic',
                'message': 'Propriétés récentes et populaires'
            }
            
        except Exception as e:
            logger.error(f"❌ Erreur generic: {e}")
            return {
                'recommendations': [],
                'total': 0,
                'is_fallback': True,
                'error': 'Service temporairement indisponible'
            }
    
    def _fast_fallback(self, limit: int) -> Dict:
        """Fallback d'urgence si timeout"""
        return {
            'recommendations': [],
            'total': 0,
            'is_fallback': True,
            'fallback_type': 'timeout',
            'message': 'Service surchargé, réessayez dans quelques secondes'
        }

# ==================== INITIALISATION ====================
supabase_url = os.getenv('SUPABASE_URL')
supabase_key = os.getenv('SUPABASE_SERVICE_ROLE_KEY')

supabase = None
recommendation_service = None

if supabase_url and supabase_key:
    try:
        supabase = create_client(supabase_url, supabase_key)
        recommendation_service = RecommendationService(supabase)
        logger.info("✅ Services initialisés")
    except Exception as e:
        logger.error(f"❌ Erreur init: {e}")
else:
    logger.warning("⚠️ Variables d'environnement manquantes")

# ==================== ROUTES OPTIMISÉES ====================
@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'ok',
        'supabase': 'connected' if supabase else 'disconnected',
        'service': 'ready' if recommendation_service else 'unavailable',
        'timestamp': datetime.now(timezone.utc).isoformat()
    })

@app.route('/recommendations', methods=['POST'])
def get_recommendations():
    if not recommendation_service:
        return jsonify({'error': 'Service indisponible', 'recommendations': []}), 503
    
    try:
        body = request.get_json() or {}
        req = RecommendationRequest(**body)
        
        # 🚀 Validation rapide
        if req.limit > Config.MAX_LIMIT:
            req.limit = Config.MAX_LIMIT
        
        result = recommendation_service.get_recommendations(req)
        
        # Nettoyer la réponse
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
        logger.exception("❌ Erreur /recommendations")
        # 🚀 Toujours retourner une réponse valide même en erreur
        return jsonify({
            'recommendations': [],
            'total': 0,
            'is_fallback': True,
            'error': 'Erreur interne',
            'message': 'Service temporairement indisponible'
        }), 500

@app.route('/feedback', methods=['POST'])
def track_feedback():
    if not supabase:
        return jsonify({'error': 'Service indisponible'}), 503
    
    try:
        body = request.get_json() or {}
        feedback = FeedbackRequest(**body)
        
        # 🚀 Fire-and-forget : ne pas attendre la réponse
        def save_feedback():
            try:
                supabase.table('feedback_events').insert({
                    'user_id': feedback.user_id,
                    'property_id': feedback.property_id,
                    'event_type': feedback.event_type,
                    'created_at': datetime.now(timezone.utc).isoformat()
                }).execute()
                recommendations_cache.invalidate(feedback.user_id)
            except Exception as e:
                logger.error(f"Erreur feedback async: {e}")
        
        threading.Thread(target=save_feedback).start()
        
        return jsonify({'success': True})
        
    except Exception as e:
        logger.exception("❌ Erreur feedback")
        return jsonify({'error': str(e)}), 500

# ==================== WARMUP POUR RENDER ====================
@app.route('/warmup', methods=['GET'])
def warmup():
    """Endpoint pour garder le service awake"""
    return jsonify({'status': 'warm', 'timestamp': time.time()})

if __name__ == '__main__':
    port = int(os.getenv('PORT', 10000))
    app.run(host='0.0.0.0', port=port, threaded=True)

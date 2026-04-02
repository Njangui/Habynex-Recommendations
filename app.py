# app.py - API de recommandations Habynex pour Render
import os
import json
import time
import hashlib
import math
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass, field
from collections import defaultdict
from functools import lru_cache

from flask import Flask, request, jsonify
from flask_cors import CORS
import redis
from supabase import create_client, Client
import numpy as np
from pydantic import BaseModel, Field, validator
import logging

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# ==================== CONFIGURATION & CONSTANTES ====================

class Config:
    CACHE_TTL = 5 * 60  # 5 minutes en secondes
    EMBEDDING_CACHE_TTL = 24 * 60 * 60  # 24h
    RATE_LIMIT = 100  # requêtes par IP
    RATE_WINDOW = 60  # 1 minute en secondes
    MAX_CANDIDATES = 150
    EMBEDDING_DIM = 48
    MMR_LAMBDA = 0.65
    DEFAULT_LIMIT = 15
    MAX_LIMIT = 50

# Schémas de validation Pydantic
class ContextSchema(BaseModel):
    source: str = Field(default='homepage', regex='^(search|homepage|alert|favorites|similar)$')
    device: str = Field(default='desktop', regex='^(mobile|desktop|tablet)$')
    urgency: Optional[str] = Field(None, regex='^(immediate|within_week|within_month|planning)$')
    referrer: Optional[str] = None
    session_id: Optional[str] = None
    ab_test_group: Optional[str] = Field(None, regex='^(control|embedding_v1|hybrid_ml|advanced)$')

class RecommendationRequestSchema(BaseModel):
    user_id: Optional[str] = None
    limit: int = Field(default=Config.DEFAULT_LIMIT, ge=1, le=Config.MAX_LIMIT)
    offset: int = Field(default=0, ge=0)
    context: Optional[ContextSchema] = None

class FeedbackEventSchema(BaseModel):
    user_id: str
    property_id: str
    event_type: str = Field(..., regex='^(view|click|favorite|contact|visit|rent)$')
    request_id: Optional[str] = None
    context: Optional[Dict[str, Any]] = None
    ab_test_group: Optional[str] = None

# Poids optimisés par variante A/B
WEIGHT_CONFIGS = {
    'control': {
        'onboarding': 35, 'viewingHistory': 25, 'collaborativeFilter': 15,
        'embeddingSimilarity': 0, 'locationMatch': 10, 'popularity': 8,
        'recency': 5, 'verification': 2, 'diversity': 10, 'mlScore': 0,
    },
    'embedding_v1': {
        'onboarding': 25, 'viewingHistory': 20, 'collaborativeFilter': 15,
        'embeddingSimilarity': 20, 'locationMatch': 10, 'popularity': 6,
        'recency': 3, 'verification': 1, 'diversity': 10, 'mlScore': 0,
    },
    'hybrid_ml': {
        'onboarding': 20, 'viewingHistory': 15, 'collaborativeFilter': 10,
        'embeddingSimilarity': 25, 'locationMatch': 8, 'popularity': 4,
        'recency': 2, 'verification': 1, 'diversity': 10, 'mlScore': 15,
    },
    'advanced': {
        'onboarding': 18, 'viewingHistory': 12, 'collaborativeFilter': 8,
        'embeddingSimilarity': 28, 'locationMatch': 6, 'popularity': 3,
        'recency': 2, 'verification': 1, 'diversity': 12, 'mlScore': 20,
    }
}

# Types de propriétés
PROPERTY_TYPE_CATEGORIES = {
    'RESIDENTIAL': [
        'studio', 'apartment', 'house', 'room', 'villa', 'duplex',
        'penthouse', 'furnished_apartment', 'shared_room'
    ],
    'LAND': ['land'],
    'COMMERCIAL': [
        'shop', 'store', 'commercial_space', 'warehouse', 'office', 'building',
        'beauty_salon', 'hair_salon', 'restaurant', 'cafe', 'bar', 'hotel',
        'pharmacy', 'clinic', 'gym', 'coworking', 'showroom', 'workshop'
    ]
}

ALL_PROPERTY_TYPES = (
    PROPERTY_TYPE_CATEGORIES['RESIDENTIAL'] +
    PROPERTY_TYPE_CATEGORIES['LAND'] +
    PROPERTY_TYPE_CATEGORIES['COMMERCIAL']
)

# Équipements complets
AMENITIES_LIST = [
    # Essential
    'wifi', 'parking', 'security', 'generator', 'water_tank',
    'electricity_prepaid', 'electricity_postpaid',
    'water_borehole', 'water_tap',
    # Comfort
    'air_conditioning', 'furnished', 'garden', 'balcony', 'terrace',
    'closet', 'water_heater', 'kitchen_cabinets', 'tiled_floor', 'ceiling_fan',
    # Services
    'pool', 'gym_amenity', 'elevator', 'cctv', 'reception', 'storage',
    'backup_generator', 'meeting_room',
    # Practical
    'pets_allowed', 'wheelchair_access', 'fence', 'gate',
    'paved_road', 'near_main_road',
    # Nearby
    'school_nearby', 'market_nearby',
    # Commercial specific
    'loading_dock', 'display_window', 'kitchen_facilities', 'bar_counter',
    'dining_area', 'alarm_system', 'fire_safety', 'handicap_access',
    'high_ceiling'
]

# ==================== DATACLASSES ====================

@dataclass
class UserProfile:
    user_id: Optional[str] = None
    city: Optional[str] = None
    budget_min: Optional[float] = None
    budget_max: Optional[float] = None
    preferred_property_types: List[str] = field(default_factory=list)
    preferred_neighborhoods: List[str] = field(default_factory=list)
    preferred_listing_types: List[str] = field(default_factory=list)
    preferred_amenities: List[str] = field(default_factory=list)
    move_in_timeline: Optional[str] = None
    user_type: Optional[str] = None
    segment: Optional[str] = None
    furnished_preference: Optional[str] = None
    parking_needed: bool = False
    pet_friendly: bool = False
    must_have_features: List[str] = field(default_factory=list)
    deal_breakers: List[str] = field(default_factory=list)
    preferred_floor: Optional[int] = None
    kitchen_type_preference: Optional[str] = None
    min_bedrooms: Optional[int] = None
    min_bathrooms: Optional[int] = None
    needs_laundry: bool = False
    needs_dining_room: bool = False

@dataclass
class ViewingPattern:
    property_types: Dict[str, int] = field(default_factory=dict)
    listing_types: Dict[str, int] = field(default_factory=dict)
    cities: Dict[str, int] = field(default_factory=dict)
    neighborhoods: Dict[str, int] = field(default_factory=dict)
    price_range: Dict[str, float] = field(default_factory=lambda: {'min': 0, 'max': 0})
    amenities: Dict[str, int] = field(default_factory=dict)
    total_views: int = 0
    avg_view_duration: float = 0
    embedding_centroid: Optional[List[float]] = None
    avg_bedrooms: Optional[float] = None
    avg_bathrooms: Optional[float] = None
    avg_area: Optional[float] = None
    furnished_ratio: Optional[float] = None

@dataclass
class ScoredProperty:
    property: Dict[str, Any]
    score: float
    reasons: List[str]
    similarity_score: float
    embedding_similarity: float
    ml_score: Optional[float] = None
    score_breakdown: Dict[str, float] = field(default_factory=dict)
    rank: Optional[int] = None

@dataclass
class FeatureWeights:
    onboarding: int
    viewing_history: int
    collaborative_filter: int
    embedding_similarity: int
    location_match: int
    popularity: int
    recency: int
    verification: int
    diversity: int
    ml_score: Optional[int] = None

# ==================== SERVICES ====================

class EmbeddingService:
    def __init__(self, supabase: Client, redis_client: redis.Redis):
        self.supabase = supabase
        self.redis = redis_client
        self.cache = {}
    
    def _get_cache_key(self, property_id: str) -> str:
        return f"emb:{property_id}"
    
    async def get_property_embeddings_batch(self, property_ids: List[str]) -> Dict[str, List[float]]:
        result = {}
        to_fetch = []
        
        # Vérifier le cache local et Redis
        for pid in property_ids:
            cache_key = self._get_cache_key(pid)
            # Cache local
            if pid in self.cache:
                cached = self.cache[pid]
                if time.time() - cached['timestamp'] < Config.EMBEDDING_CACHE_TTL:
                    result[pid] = cached['vector']
                    continue
            
            # Cache Redis
            try:
                cached_redis = self.redis.get(cache_key)
                if cached_redis:
                    vector = json.loads(cached_redis)
                    result[pid] = vector
                    self.cache[pid] = {'vector': vector, 'timestamp': time.time()}
                    continue
            except Exception as e:
                logger.error(f"Redis cache error: {e}")
            
            to_fetch.append(pid)
        
        if not to_fetch:
            return result
        
        # Récupérer depuis Supabase
        try:
            response = self.supabase.table('property_embeddings')\
                .select('property_id, vector')\
                .in_('property_id', to_fetch)\
                .execute()
            
            for row in response.data:
                pid = row['property_id']
                vector = row['vector']
                result[pid] = vector
                
                # Mettre en cache
                self.cache[pid] = {'vector': vector, 'timestamp': time.time()}
                try:
                    self.redis.setex(
                        self._get_cache_key(pid),
                        Config.EMBEDDING_CACHE_TTL,
                        json.dumps(vector)
                    )
                except Exception as e:
                    logger.error(f"Redis set error: {e}")
                    
        except Exception as e:
            logger.error(f"Batch embedding fetch error: {e}")
        
        return result
    
    async def get_user_embedding(self, user_id: str) -> Optional[List[float]]:
        try:
            response = self.supabase.table('user_embeddings')\
                .select('vector')\
                .eq('user_id', user_id)\
                .single()\
                .execute()
            
            if response.data:
                return response.data['vector']
        except Exception as e:
            logger.error(f"Get user embedding error: {e}")
        
        return None
    
    async def create_user_embedding(self, user_id: str, viewed_properties: List[Dict]) -> List[float]:
        if not viewed_properties:
            return [0.0] * Config.EMBEDDING_DIM
        
        property_ids = [p['id'] for p in viewed_properties]
        embeddings_map = await self.get_property_embeddings_batch(property_ids)
        valid_embeddings = list(embeddings_map.values())
        
        if not valid_embeddings:
            return [0.0] * Config.EMBEDDING_DIM
        
        centroid = self._calculate_centroid(valid_embeddings)
        
        # Sauvegarder dans Supabase
        try:
            self.supabase.table('user_embeddings').upsert({
                'user_id': user_id,
                'vector': centroid,
                'version': 'v3',
                'property_count': len(valid_embeddings),
                'updated_at': datetime.utcnow().isoformat()
            }).execute()
        except Exception as e:
            logger.error(f"Error saving user embedding: {e}")
        
        return centroid
    
    def _calculate_centroid(self, embeddings: List[List[float]]) -> List[float]:
        dim = len(embeddings[0])
        centroid = [0.0] * dim
        count = len(embeddings)
        
        for emb in embeddings:
            for i, val in enumerate(emb):
                centroid[i] += val / count
        
        return centroid
    
    @staticmethod
    def cosine_similarity(a: List[float], b: List[float]) -> float:
        if not a or not b or len(a) != len(b):
            return 0.0
        
        dot_product = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(x * x for x in b))
        
        if norm_a == 0 or norm_b == 0:
            return 0.0
        
        return dot_product / (norm_a * norm_b)
    
    def compute_property_embedding(self, property: Dict[str, Any]) -> List[float]:
        vector = []
        
        # 1. PRIX (normalisé)
        vector.append(math.log1p(property.get('price', 0)) / 15)
        vector.append(1.0 if property.get('price_unit') == 'month' else 0.0)
        vector.append(1.0 if property.get('price_unit') == 'day' else 0.0)
        vector.append(1.0 if property.get('price_unit') == 'sale' else 0.0)
        
        # 2. TYPES DE PROPRIÉTÉ
        residential_types = ['studio', 'apartment', 'house', 'room', 'villa', 'duplex', 'penthouse', 'furnished_apartment', 'shared_room']
        for ptype in residential_types:
            vector.append(1.0 if property.get('property_type') == ptype else 0.0)
        
        # Terrain
        vector.append(1.0 if property.get('property_type') == 'land' else 0.0)
        
        # Commercial de base
        commercial_types = ['shop', 'store', 'commercial_space', 'warehouse', 'office', 'building']
        for ptype in commercial_types:
            vector.append(1.0 if property.get('property_type') == ptype else 0.0)
        
        # Nouveaux commerces spécifiques
        specific_commercial = ['beauty_salon', 'hair_salon', 'restaurant', 'cafe', 'bar', 'hotel', 'pharmacy', 'clinic', 'gym', 'coworking', 'showroom', 'workshop']
        for ptype in specific_commercial:
            vector.append(1.0 if property.get('property_type') == ptype else 0.0)
        
        # 3. TYPES D'ANNONCE
        vector.append(1.0 if property.get('listing_type') == 'rent' else 0.0)
        vector.append(1.0 if property.get('listing_type') == 'sale' else 0.0)
        vector.append(1.0 if property.get('listing_type') == 'colocation' else 0.0)
        vector.append(1.0 if property.get('listing_type') == 'short_term' else 0.0)
        
        # 4. LOCALISATION (hash normalisé)
        vector.append(self._hash_string(property.get('city', '')) / 10000)
        vector.append(self._hash_string(property.get('neighborhood', '')) / 10000)
        vector.append(self._hash_string(property.get('address', '')) / 10000)
        
        # 5. CARACTÉRISTIQUES PHYSIQUES
        vector.append(min(property.get('area', 0) / 500, 1.0))
        vector.append(min(property.get('bedrooms', 0) / 5, 1.0))
        vector.append(min(property.get('bathrooms', 0) / 3, 1.0))
        vector.append(min(property.get('living_rooms', 0) / 3, 1.0))
        vector.append(min(property.get('kitchens', 0) / 2, 1.0))
        vector.append(min(property.get('dining_rooms', 0) / 2, 1.0))
        vector.append(min(property.get('laundry_rooms', 0) / 2, 1.0))
        
        # 6. ÉTAGES
        vector.append(min(property.get('floor', 0) / 10, 1.0))
        vector.append(min(property.get('total_floors', 0) / 20, 1.0))
        vector.append(1.0 if property.get('kitchen_type') == 'open' else 0.0)
        vector.append(1.0 if property.get('kitchen_type') == 'closed' else 0.0)
        
        # 7. ÉQUIPEMENTS
        all_amenities = [
            'wifi', 'parking', 'security', 'generator', 'water_tank',
            'electricity_prepaid', 'electricity_postpaid',
            'water_borehole', 'water_tap',
            'air_conditioning', 'furnished', 'garden', 'balcony', 'terrace',
            'closet', 'water_heater', 'kitchen_cabinets', 'tiled_floor', 'ceiling_fan',
            'pool', 'gym', 'elevator', 'cctv', 'reception', 'storage',
            'backup_generator', 'meeting_room',
            'fence', 'gate', 'paved_road', 'near_main_road',
            'school_nearby', 'market_nearby',
            'loading_dock', 'display_window', 'kitchen_facilities', 'bar_counter',
            'dining_area', 'alarm_system', 'fire_safety', 'handicap_access', 'high_ceiling'
        ]
        
        property_amenities = property.get('amenities', [])
        for amenity in all_amenities:
            has_amenity = amenity in property_amenities or self._map_amenity_name(amenity) in property_amenities
            vector.append(1.0 if has_amenity else 0.0)
        
        # 8. ÉTAT ET MÉTADONNÉES
        vector.append(1.0 if property.get('is_furnished') else 0.0)
        vector.append(1.0 if property.get('is_agent_verified') else 0.0)
        vector.append(1.0 if property.get('whatsapp_enabled') else 0.0)
        
        # 9. IMAGES ET CONTENU
        images = property.get('images', [])
        vector.append(min(len(images) / 10, 1.0))
        desc = property.get('description', '')
        vector.append(min(len(desc) / 2000, 1.0) if desc else 0.0)
        
        # 10. POPULARITÉ
        vector.append(math.log1p(property.get('view_count', 0)) / 10)
        vector.append(math.log1p(property.get('favorite_count', 0)) / 10)
        vector.append(math.log1p(property.get('contact_count', 0)) / 10)
        
        # 11. TEMPS
        vector.append(self._days_since(property.get('created_at')) / 60)
        vector.append(1.0 if property.get('is_available') else 0.0)
        vector.append(self._days_until(property.get('available_from')) / 30)
        
        # 12. DÉPÔT DE GARANTIE
        deposit = property.get('deposit')
        vector.append(math.log1p(deposit) / 15 if deposit else 0.0)
        
        # Compléter ou tronquer à EMBEDDING_DIM
        while len(vector) < Config.EMBEDDING_DIM:
            vector.append(0.0)
        
        return vector[:Config.EMBEDDING_DIM]
    
    def _map_amenity_name(self, amenity: str) -> str:
        mapping = {
            'gym': 'gym_amenity',
            'pets_allowed': 'pets'
        }
        return mapping.get(amenity, amenity)
    
    @staticmethod
    def _hash_string(s: str) -> int:
        hash_val = 0
        for char in s:
            hash_val = ((hash_val << 5) - hash_val) + ord(char)
            hash_val = hash_val & 0xFFFFFFFF
        return abs(hash_val)
    
    @staticmethod
    def _days_since(date_str: Optional[str]) -> float:
        if not date_str:
            return 30.0
        try:
            date = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
            return (datetime.utcnow() - date).total_seconds() / (24 * 3600)
        except:
            return 30.0
    
    @staticmethod
    def _days_until(date_str: Optional[str]) -> float:
        if not date_str:
            return 0.0
        try:
            date = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
            return max(0, (date - datetime.utcnow()).total_seconds() / (24 * 3600))
        except:
            return 0.0

class ABTestingFramework:
    def __init__(self, supabase: Client, redis_client: redis.Redis):
        self.supabase = supabase
        self.redis = redis_client
        self.active_experiments = {}
        self._load_experiments()
    
    def _load_experiments(self):
        try:
            response = self.supabase.table('ab_experiments')\
                .select('*')\
                .eq('status', 'active')\
                .execute()
            
            for exp in response.data:
                self.active_experiments[exp['id']] = exp
        except Exception as e:
            logger.error(f"Failed to load experiments: {e}")
    
    def assign_group(self, user_id: str, experiment_id: str = 'default') -> str:
        experiment = self.active_experiments.get(experiment_id)
        if not experiment:
            return 'control'
        
        # Hash simple pour déterminisme
        hash_val = sum(ord(c) for c in user_id)
        variants = experiment.get('variants', ['control', 'treatment'])
        return variants[hash_val % len(variants)]
    
    async def track_event(self, event: Dict[str, Any]):
        try:
            event['timestamp'] = datetime.utcnow().isoformat()
            self.supabase.table('ab_events').insert(event).execute()
        except Exception as e:
            logger.error(f"AB tracking error: {e}")

class LightMLModel:
    def __init__(self, supabase: Client):
        self.supabase = supabase
        self.weights = []
        self.bias = 0.0
        self.is_loaded = False
        self._load_weights()
    
    def _load_weights(self):
        try:
            response = self.supabase.table('ml_model_weights')\
                .select('weights, bias')\
                .eq('model_name', 'light_ranker_v3')\
                .single()\
                .execute()
            
            if response.data:
                self.weights = response.data['weights']
                self.bias = response.data.get('bias', 0)
                self.is_loaded = True
        except Exception as e:
            logger.error(f"Failed to load ML weights: {e}")
    
    def predict(self, features: List[float]) -> float:
        if not self.is_loaded or len(self.weights) != len(features):
            return 0.5
        
        score = self.bias
        for i, feat in enumerate(features):
            score += feat * self.weights[i]
        
        # Sigmoid
        return 1 / (1 + math.exp(-score))
    
    def extract_features(self, property: Dict, user_profile: Optional[UserProfile], context: Optional[Dict]) -> List[float]:
        features = [
            # Prix
            min(property.get('price', 0) / 10000, 5),
            (property.get('price', 0) - (user_profile.budget_min or 0)) / 5000 if user_profile and user_profile.budget_min else 0,
            ((user_profile.budget_max or 0) - property.get('price', 0)) / 5000 if user_profile and user_profile.budget_max else 0,
            
            # Caractéristiques
            min(property.get('area', 0) / 200, 3),
            min(property.get('bedrooms', 0) / 5, 1),
            min(property.get('bathrooms', 0) / 3, 1),
            min(property.get('living_rooms', 0) / 3, 1),
            
            # Localisation
            1.0 if user_profile and user_profile.city == property.get('city') else 0.0,
            1.0 if user_profile and property.get('neighborhood') in user_profile.preferred_neighborhoods else 0.0,
            
            # Type match
            1.0 if user_profile and property.get('property_type') in user_profile.preferred_property_types else 0.0,
            
            # Équipements clés
            1.0 if (user_profile and user_profile.furnished_preference == 'furnished' and property.get('is_furnished')) else 0.0,
            1.0 if (user_profile and user_profile.parking_needed and 'parking' in property.get('amenities', [])) else 0.0,
            1.0 if (user_profile and user_profile.pet_friendly and 'pets_allowed' in property.get('amenities', [])) else 0.0,
            
            # Contexte
            1.0 if context and context.get('device') == 'mobile' else 0.0,
            1.0 if context and context.get('urgency') == 'immediate' else 0.0,
            
            # Qualité
            1.0 if property.get('is_agent_verified') else 0.0,
            math.log1p(property.get('view_count', 0)) / 10,
            min(len(property.get('images', [])) / 10, 1),
            
            # Spécifiques CreateListing
            1.0 if (user_profile and property.get('kitchen_type') == user_profile.kitchen_type_preference) else 0.0,
            1.0 if (user_profile and property.get('bedrooms', 0) >= (user_profile.min_bedrooms or 0)) else 0.0,
            1.0 if (user_profile and property.get('bathrooms', 0) >= (user_profile.min_bathrooms or 0)) else 0.0,
        ]
        
        return features

class DistributedCache:
    def __init__(self, redis_client: redis.Redis):
        self.local_cache = {}
        self.request_counts = {}
        self.redis = redis_client
    
    def get(self, key: str) -> Optional[Any]:
        # Cache local
        if key in self.local_cache:
            item = self.local_cache[key]
            if time.time() - item['timestamp'] < Config.CACHE_TTL:
                item['hits'] += 1
                return item['data']
            else:
                del self.local_cache[key]
        
        # Cache Redis
        try:
            data = self.redis.get(key)
            if data:
                parsed = json.loads(data)
                self.local_cache[key] = {'data': parsed, 'timestamp': time.time(), 'hits': 1}
                return parsed
        except Exception as e:
            logger.error(f"Redis get error: {e}")
        
        return None
    
    def set(self, key: str, data: Any, ttl: int = Config.CACHE_TTL):
        self.local_cache[key] = {'data': data, 'timestamp': time.time(), 'hits': 0}
        try:
            self.redis.setex(key, ttl, json.dumps(data))
        except Exception as e:
            logger.error(f"Redis set error: {e}")
    
    def check_rate_limit(self, ip: str) -> bool:
        now = time.time()
        record = self.request_counts.get(ip)
        
        if not record or now > record['reset_time']:
            self.request_counts[ip] = {'count': 1, 'reset_time': now + Config.RATE_WINDOW}
            return True
        
        if record['count'] >= Config.RATE_LIMIT:
            return False
        
        record['count'] += 1
        return True
    
    def get_stats(self) -> Dict[str, Any]:
        stats = {}
        for key, item in self.local_cache.items():
            stats[key] = {
                'age': time.time() - item['timestamp'],
                'hits': item['hits']
            }
        return stats

# ==================== SCORING ENGINE ====================

class ScoringEngine:
    def __init__(self, embedding_service: EmbeddingService, light_ml_model: Optional[LightMLModel] = None):
        self.embedding_service = embedding_service
        self.light_ml_model = light_ml_model
    
    def calculate_onboarding_score(self, property: Dict, user_profile: UserProfile) -> Tuple[float, List[str], Dict[str, float]]:
        score = 0.0
        reasons = []
        breakdown = {}
        
        if not user_profile:
            return score, reasons, breakdown
        
        # 1. BUDGET (critique)
        if user_profile.budget_min is not None and user_profile.budget_max is not None:
            price = property.get('price', 0)
            if user_profile.budget_min <= price <= user_profile.budget_max:
                pts = 20.0
                score += pts
                breakdown['budget'] = pts
                reasons.append("onboarding_budget_match")
            elif price < user_profile.budget_min * 0.9:
                pts = 8.0
                score += pts
                breakdown['budget_below'] = pts
            else:
                score -= 15
                breakdown['budget_penalty'] = -15
        
        # 2. TYPE DE PROPRIÉTÉ
        if property.get('property_type') in user_profile.preferred_property_types:
            pts = 10.0
            score += pts
            breakdown['property_type'] = pts
            reasons.append(f"type_{property['property_type']}")
        
        # 3. LOCALISATION
        if user_profile.city and property.get('city', '').lower() == user_profile.city.lower():
            pts = 12.0
            score += pts
            breakdown['city'] = pts
            reasons.append("onboarding_city")
        
        if property.get('neighborhood') in user_profile.preferred_neighborhoods:
            pts = 15.0
            score += pts
            breakdown['neighborhood'] = pts
            reasons.append("onboarding_neighborhood")
        
        # 4. TYPE D'ANNONCE
        if property.get('listing_type') in user_profile.preferred_listing_types:
            pts = 8.0
            score += pts
            breakdown['listing_type'] = pts
            reasons.append(f"listing_{property['listing_type']}")
        
        # 5. ÉQUIPEMENTS
        if user_profile.preferred_amenities and property.get('amenities'):
            matches = sum(1 for a in user_profile.preferred_amenities 
                         if a in property['amenities'] or self._normalize_amenity(a) in property['amenities'])
            ratio = matches / len(user_profile.preferred_amenities)
            pts = ratio * 18
            score += pts
            breakdown['amenities'] = pts
            if ratio > 0.7:
                reasons.append("onboarding_amenities_strong")
            elif ratio > 0.4:
                reasons.append("onboarding_amenities_good")
        
        # 6. TIMELINE
        if user_profile.move_in_timeline == "immediate" and property.get('is_available'):
            pts = 10.0
            score += pts
            breakdown['timeline'] = pts
            reasons.append("onboarding_immediate")
        
        # 7. MEUBLÉ
        if user_profile.furnished_preference and user_profile.furnished_preference != 'either':
            if (user_profile.furnished_preference == 'furnished' and property.get('is_furnished')) or \
               (user_profile.furnished_preference == 'unfurnished' and not property.get('is_furnished')):
                pts = 8.0
                score += pts
                breakdown[f'{user_profile.furnished_preference}_match'] = pts
                reasons.append(f"{user_profile.furnished_preference}_match")
        
        # 8. PARKING
        if user_profile.parking_needed:
            if any('parking' in a for a in property.get('amenities', [])):
                pts = 7.0
                score += pts
                breakdown['parking_match'] = pts
                reasons.append("parking_available")
            else:
                score -= 6
                breakdown['parking_penalty'] = -6
        
        # 9. ANIMAUX
        if user_profile.pet_friendly:
            if any('pet' in a or 'animal' in a for a in property.get('amenities', [])):
                pts = 9.0
                score += pts
                breakdown['pets_allowed'] = pts
                reasons.append("pets_welcome")
        
        # 10. ÉTAGE
        if user_profile.preferred_floor is not None and property.get('floor') is not None:
            floor_diff = abs(property['floor'] - user_profile.preferred_floor)
            if floor_diff == 0:
                pts = 6.0
                score += pts
                breakdown['floor_exact'] = pts
                reasons.append("preferred_floor")
            elif floor_diff <= 2:
                pts = 3.0
                score += pts
                breakdown['floor_close'] = pts
        
        # 11. CUISINE
        if user_profile.kitchen_type_preference and property.get('kitchen_type'):
            if property['kitchen_type'] == user_profile.kitchen_type_preference:
                pts = 5.0
                score += pts
                breakdown['kitchen_type'] = pts
                reasons.append(f"kitchen_{property['kitchen_type']}")
        
        # 12. CHAMBRES/SALLES DE BAIN MINIMUM
        if user_profile.min_bedrooms and property.get('bedrooms', 0) >= user_profile.min_bedrooms:
            pts = 4.0
            score += pts
            breakdown['min_bedrooms'] = pts
        
        if user_profile.min_bathrooms and property.get('bathrooms', 0) >= user_profile.min_bathrooms:
            pts = 4.0
            score += pts
            breakdown['min_bathrooms'] = pts
        
        # 13. MUST-HAVE FEATURES
        if user_profile.must_have_features and property.get('amenities'):
            has_all = all(f in property['amenities'] or self._normalize_amenity(f) in property['amenities'] 
                         for f in user_profile.must_have_features)
            if has_all:
                pts = 12.0
                score += pts
                breakdown['must_haves'] = pts
                reasons.append("all_must_haves")
            else:
                missing = sum(1 for f in user_profile.must_have_features 
                             if f not in property['amenities'] and self._normalize_amenity(f) not in property['amenities'])
                score -= missing * 4
                breakdown['must_haves_missing'] = -missing * 4
        
        # 14. DEAL BREAKERS
        if user_profile.deal_breakers and property.get('amenities'):
            has_deal_breaker = any(db in property['amenities'] or self._normalize_amenity(db) in property['amenities'] 
                                  for db in user_profile.deal_breakers)
            if has_deal_breaker:
                score -= 20
                breakdown['deal_breaker'] = -20
                reasons.append("deal_breaker_present")
        
        # 15. BUANDERIE/SALLE À MANGER
        if user_profile.needs_laundry and property.get('laundry_rooms', 0) > 0:
            pts = 4.0
            score += pts
            breakdown['laundry'] = pts
        
        if user_profile.needs_dining_room and property.get('dining_rooms', 0) > 0:
            pts = 4.0
            score += pts
            breakdown['dining_room'] = pts
        
        # 16. SEGMENT UTILISATEUR
        if user_profile.segment == "student" and property.get('price', 0) < 400000:
            score += 6
            breakdown['segment'] = 6
            reasons.append("segment_student")
        elif user_profile.segment == "family" and property.get('bedrooms', 0) >= 3:
            score += 7
            breakdown['segment'] = 7
            reasons.append("segment_family")
        elif user_profile.segment == "expat" and property.get('is_furnished'):
            score += 6
            breakdown['segment'] = 6
            reasons.append("segment_expat")
        elif user_profile.segment == "commercial" and property.get('property_type') in PROPERTY_TYPE_CATEGORIES['COMMERCIAL']:
            score += 8
            breakdown['segment'] = 8
            reasons.append("segment_commercial")
        
        return score, reasons, breakdown
    
    def _normalize_amenity(self, amenity: str) -> str:
        normalized = amenity.lower().replace('_', '').replace('-', '')
        mappings = {
            'wifi': 'wifi',
            'parking': 'parking',
            'garage': 'parking',
            'climatisation': 'air_conditioning',
            'ac': 'air_conditioning',
            'meuble': 'furnished',
            'meublé': 'furnished',
            'animaux': 'pets_allowed',
            'pets': 'pets_allowed',
        }
        return mappings.get(normalized, amenity)
    
    def calculate_property_score(
        self,
        property: Dict,
        user_profile: Optional[UserProfile],
        viewing_pattern: Optional[ViewingPattern],
        collaborative_scores: Dict[str, float],
        user_favorites: List[str],
        weights: FeatureWeights,
        context: Optional[Dict],
        user_embedding: Optional[List[float]]
    ) -> ScoredProperty:
        score = 0.0
        reasons = []
        breakdown = {}
        similarity_score = 0.0
        embedding_similarity = 0.0
        ml_score = 0.5
        
        # 1. Onboarding Score
        if user_profile:
            onboarding_score, onboarding_reasons, onboarding_breakdown = self.calculate_onboarding_score(property, user_profile)
            contribution = min(onboarding_score, 70) * (weights.onboarding / 35)
            score += contribution
            breakdown['onboarding'] = contribution
            reasons.extend(onboarding_reasons[:4])
        
        # 2. Embedding Similarity
        if user_embedding and property.get('property_embeddings', {}).get('vector'):
            embedding_similarity = self.embedding_service.cosine_similarity(
                user_embedding, 
                property['property_embeddings']['vector']
            )
            if weights.embedding_similarity:
                contribution = embedding_similarity * weights.embedding_similarity
                score += contribution
                breakdown['embedding_similarity'] = contribution
                if embedding_similarity > 0.8:
                    reasons.append("high_embedding_match")
                elif embedding_similarity > 0.6:
                    reasons.append("good_embedding_match")
        
        # 3. Viewing History Pattern
        if viewing_pattern and viewing_pattern.embedding_centroid and property.get('property_embeddings', {}).get('vector'):
            similarity_score = self.embedding_service.cosine_similarity(
                viewing_pattern.embedding_centroid,
                property['property_embeddings']['vector']
            )
            contribution = similarity_score * weights.viewing_history
            score += contribution
            breakdown['viewing_history'] = contribution
            if similarity_score > 0.7:
                reasons.append("matches_viewing_history")
        
        # 4. Collaborative Filtering
        collab_score = collaborative_scores.get(property['id'])
        if collab_score:
            normalized_collab = min(collab_score / 5, 1.0)
            contribution = normalized_collab * weights.collaborative_filter
            score += contribution
            breakdown['collaborative'] = contribution
            if normalized_collab > 0.5:
                reasons.append("trending_with_similar_users")
        
        # 5. ML Score
        if self.light_ml_model and weights.ml_score:
            features = self.light_ml_model.extract_features(property, user_profile, context)
            ml_score = self.light_ml_model.predict(features)
            contribution = ml_score * weights.ml_score
            score += contribution
            breakdown['ml_score'] = contribution
            if ml_score > 0.8:
                reasons.append("ml_high_confidence")
        
        # 6. Déjà favori
        if property['id'] in user_favorites:
            score -= 30
            breakdown['already_favorite'] = -30
        
        # 7. Popularité avec décroissance temporelle
        days_since_created = self._days_since(property.get('created_at'))
        recency_weight = max(0.3, 1 - (days_since_created / 30))
        popularity_score = math.log10(property.get('view_count', 0) + 1) * recency_weight
        capped_popularity = min(popularity_score, weights.popularity)
        score += capped_popularity
        breakdown['popularity'] = capped_popularity
        
        # 8. Boost récence
        if days_since_created < 7:
            recency_score = ((7 - days_since_created) / 7) * weights.recency
            score += recency_score
            breakdown['recency'] = recency_score
            if days_since_created < 2:
                reasons.append("new_listing")
        
        # 9. Vérification agent
        if property.get('is_agent_verified'):
            score += weights.verification
            breakdown['verification'] = weights.verification
            reasons.append("verified_agent")
        
        # 10. Urgence contextuelle
        if context and context.get('urgency') == 'immediate' and property.get('is_available'):
            days_until = self._days_until(property.get('available_from'))
            if days_until <= 7:
                score += 6
                breakdown['urgency'] = 6
                reasons.append("immediate_availability")
        
        return ScoredProperty(
            property=property,
            score=round(score, 2),
            reasons=reasons[:5],
            similarity_score=similarity_score,
            embedding_similarity=embedding_similarity,
            ml_score=ml_score,
            score_breakdown=breakdown
        )
    
    @staticmethod
    def _days_since(date_str: Optional[str]) -> float:
        if not date_str:
            return 30.0
        try:
            date = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
            return (datetime.utcnow() - date).total_seconds() / (24 * 3600)
        except:
            return 30.0
    
    @staticmethod
    def _days_until(date_str: Optional[str]) -> float:
        if not date_str:
            return 0.0
        try:
            date = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
            return max(0, (date - datetime.utcnow()).total_seconds() / (24 * 3600))
        except:
            return 0.0

# ==================== DIVERSIFICATION ENGINE ====================

class DiversificationEngine:
    def apply_mmr(self, candidates: List[ScoredProperty], limit: int, lambda_param: float = Config.MMR_LAMBDA) -> List[ScoredProperty]:
        if len(candidates) <= limit:
            for i, c in enumerate(candidates):
                c.rank = i + 1
            return candidates
        
        selected = []
        remaining = sorted(candidates, key=lambda x: x.score, reverse=True)
        
        while len(selected) < limit and remaining:
            max_mmr_score = float('-inf')
            max_mmr_index = 0
            
            for i, item in enumerate(remaining):
                if not selected:
                    mmr_score = item.score
                else:
                    max_sim = max(self._calculate_similarity(item.property, sel.property) for sel in selected)
                    mmr_score = lambda_param * item.score - (1 - lambda_param) * max_sim * 100
                
                if mmr_score > max_mmr_score:
                    max_mmr_score = mmr_score
                    max_mmr_index = i
            
            selected.append(remaining[max_mmr_index])
            remaining.pop(max_mmr_index)
        
        for i, item in enumerate(selected):
            item.rank = i + 1
        
        return selected
    
    def _calculate_similarity(self, a: Dict, b: Dict) -> float:
        similarity = 0.0
        features = 0
        
        if a.get('property_type') == b.get('property_type'):
            similarity += 1
            features += 1
        
        if a.get('city') == b.get('city'):
            similarity += 1
            features += 1
        
        if a.get('neighborhood') == b.get('neighborhood'):
            similarity += 1
            features += 1
        
        if a.get('price') and b.get('price'):
            max_price = max(a['price'], b['price'])
            if max_price > 0:
                similarity += 1 - min(abs(a['price'] - b['price']) / max_price, 1)
                features += 1
        
        if a.get('property_embeddings', {}).get('vector') and b.get('property_embeddings', {}).get('vector'):
            similarity += self._cosine_similarity(
                a['property_embeddings']['vector'],
                b['property_embeddings']['vector']
            )
            features += 1
        
        return similarity / features if features > 0 else 0.0
    
    @staticmethod
    def _cosine_similarity(a: List[float], b: List[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(x * x for x in b))
        return dot / (norm_a * norm_b + 1e-8)

# ==================== UTILITAIRES ====================

def analyze_viewing_pattern(viewed_props: List[Dict]) -> ViewingPattern:
    property_types = defaultdict(int)
    listing_types = defaultdict(int)
    cities = defaultdict(int)
    neighborhoods = defaultdict(int)
    amenities_count = defaultdict(int)
    prices = []
    bedrooms = []
    bathrooms = []
    areas = []
    total_duration = 0
    furnished_count = 0
    embeddings = []
    
    for p in viewed_props:
        property_types[p['property_type']] += 1
        listing_types[p['listing_type']] += 1
        cities[p['city']] += 1
        if p.get('neighborhood'):
            neighborhoods[p['neighborhood']] += 1
        prices.append(p['price'])
        if p.get('bedrooms'):
            bedrooms.append(p['bedrooms'])
        if p.get('bathrooms'):
            bathrooms.append(p['bathrooms'])
        if p.get('area'):
            areas.append(p['area'])
        if p.get('is_furnished'):
            furnished_count += 1
        total_duration += p.get('view_duration_seconds', 0) or 0
        
        if p.get('property_embeddings', {}).get('vector'):
            embeddings.append(p['property_embeddings']['vector'])
        
        if p.get('amenities'):
            for a in p['amenities']:
                amenities_count[a] += 1
    
    embedding_centroid = None
    if embeddings:
        dim = len(embeddings[0])
        centroid = [0.0] * dim
        for emb in embeddings:
            for i, val in enumerate(emb):
                centroid[i] += val / len(embeddings)
        embedding_centroid = centroid
    
    return ViewingPattern(
        property_types=dict(property_types),
        listing_types=dict(listing_types),
        cities=dict(cities),
        neighborhoods=dict(neighborhoods),
        price_range={
            'min': min(prices) * 0.8 if prices else 0,
            'max': max(prices) * 1.2 if prices else 0
        },
        amenities=dict(amenities_count),
        total_views=len(viewed_props),
        avg_view_duration=total_duration / len(viewed_props) if viewed_props else 0,
        embedding_centroid=embedding_centroid,
        avg_bedrooms=sum(bedrooms) / len(bedrooms) if bedrooms else None,
        avg_bathrooms=sum(bathrooms) / len(bathrooms) if bathrooms else None,
        avg_area=sum(areas) / len(areas) if areas else None,
        furnished_ratio=furnished_count / len(viewed_props) if viewed_props else None
    )

async def get_collaborative_scores(supabase: Client, user_id: Optional[str], user_favorites: List[str]) -> Dict[str, float]:
    scores = {}
    if not user_id or not user_favorites:
        return scores
    
    try:
        # Utilisateurs similaires
        response = supabase.table('property_favorites')\
            .select('user_id, property_id')\
            .in_('property_id', user_favorites[:20])\
            .neq('user_id', user_id)\
            .limit(500)\
            .execute()
        
        if not response.data:
            return scores
        
        # Matrice utilisateur-item
        user_item_matrix = defaultdict(set)
        for row in response.data:
            user_item_matrix[row['user_id']].add(row['property_id'])
        
        target_items = set(user_favorites)
        similarities = []
        
        for other_user_id, items in user_item_matrix.items():
            intersection = target_items & items
            if intersection:
                similarity = len(intersection) / math.sqrt(len(target_items) * len(items))
                if similarity > 0.2:
                    similarities.append((other_user_id, similarity))
        
        if not similarities:
            return scores
        
        # Top utilisateurs similaires
        top_users = [uid for uid, _ in sorted(similarities, key=lambda x: x[1], reverse=True)[:10]]
        
        # Recommandations
        recs_response = supabase.table('property_favorites')\
            .select('property_id, user_id')\
            .in_('user_id', top_users)\
            .not_.in_('property_id', list(target_items)[:100])\
            .limit(300)\
            .execute()
        
        if recs_response.data:
            item_scores = defaultdict(float)
            for rec in recs_response.data:
                user_sim = next((sim for uid, sim in similarities if uid == rec['user_id']), 0)
                item_scores[rec['property_id']] += user_sim
            
            # Top 100
            sorted_items = sorted(item_scores.items(), key=lambda x: x[1], reverse=True)[:100]
            for pid, score in sorted_items:
                scores[pid] = score
                
    except Exception as e:
        logger.error(f"Collaborative filtering error: {e}")
    
    return scores

async def generate_candidates(
    supabase: Client,
    user_profile: Optional[UserProfile],
    limit: int,
    excluded_ids: List[str] = None
) -> List[Dict]:
    excluded_ids = excluded_ids or []
    
    try:
        query = supabase.table('properties')\
            .select('*, property_embeddings(vector)')\
            .eq('is_published', True)\
            .eq('is_available', True)
        
        if excluded_ids:
            # Supabase Python ne supporte pas not.in directement de la même façon
            # On filtre après coup pour simplifier
            pass
        
        # Visiteurs sans profil
        if not user_profile:
            response = query.limit(min(limit * 5, Config.MAX_CANDIDATES)).execute()
            
            if not response.data:
                return []
            
            scored = []
            for p in response.data:
                if p['id'] in excluded_ids:
                    continue
                    
                days_since = (datetime.utcnow() - datetime.fromisoformat(p['created_at'].replace('Z', '+00:00'))).days if p.get('created_at') else 30
                recency_score = max(0, 1 - (days_since / 30))
                popularity_score = math.log1p(p.get('view_count', 0)) / 5
                
                p['_anonymous_score'] = recency_score * 0.7 + popularity_score * 0.3
                scored.append(p)
            
            scored.sort(key=lambda x: x['_anonymous_score'], reverse=True)
            return scored[:min(limit * 3, Config.MAX_CANDIDATES)]
        
        # Utilisateurs avec profil
        if user_profile.city:
            query = query.ilike('city', f'%{user_profile.city}%')
        
        if user_profile.budget_min is not None and user_profile.budget_max is not None:
            query = query.gte('price', user_profile.budget_min * 0.8).lte('price', user_profile.budget_max * 1.2)
        
        if user_profile.preferred_property_types:
            query = query.in_('property_type', user_profile.preferred_property_types)
        
        if user_profile.preferred_neighborhoods:
            query = query.in_('neighborhood', user_profile.preferred_neighborhoods)
        
        # Filtres spécifiques CreateListing
        if user_profile.furnished_preference == 'furnished':
            query = query.eq('is_furnished', True)
        elif user_profile.furnished_preference == 'unfurnished':
            query = query.eq('is_furnished', False)
        
        if user_profile.parking_needed:
            query = query.contains('amenities', ['parking'])
        
        if user_profile.pet_friendly:
            query = query.contains('amenities', ['pets_allowed'])
        
        if user_profile.preferred_floor is not None:
            query = query.gte('floor', user_profile.preferred_floor - 1).lte('floor', user_profile.preferred_floor + 1)
        
        if user_profile.kitchen_type_preference:
            query = query.eq('kitchen_type', user_profile.kitchen_type_preference)
        
        if user_profile.min_bedrooms:
            query = query.gte('bedrooms', user_profile.min_bedrooms)
        
        if user_profile.min_bathrooms:
            query = query.gte('bathrooms', user_profile.min_bathrooms)
        
        if user_profile.needs_laundry:
            query = query.gt('laundry_rooms', 0)
        
        if user_profile.needs_dining_room:
            query = query.gt('dining_rooms', 0)
        
        response = query.order('created_at', desc=True).limit(min(limit * 3, Config.MAX_CANDIDATES)).execute()
        
        # Filtrer les exclus manuellement
        data = [p for p in (response.data or []) if p['id'] not in excluded_ids]
        return data
        
    except Exception as e:
        logger.error(f"Candidate generation error: {e}")
        return []

# ==================== ROUTES ====================

# Initialisation des clients
redis_client = redis.Redis.from_url(os.getenv('REDIS_URL', 'redis://localhost:6379/0'), decode_responses=True)
supabase_url = os.getenv('SUPABASE_URL')
supabase_key = os.getenv('SUPABASE_SERVICE_ROLE_KEY')

if not supabase_url or not supabase_key:
    raise ValueError("Missing Supabase environment variables")

supabase: Client = create_client(supabase_url, supabase_key)
global_cache = DistributedCache(redis_client)

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy', 'timestamp': datetime.utcnow().isoformat()})

@app.route('/recommendations', methods=['POST', 'OPTIONS'])
async def recommendations():
    if request.method == 'OPTIONS':
        return '', 204
    
    request_start_time = time.time() * 1000
    
    # Rate limiting
    client_ip = request.headers.get('x-forwarded-for', request.remote_addr) or 'unknown'
    if not global_cache.check_rate_limit(client_ip):
        return jsonify({'error': 'Rate limit exceeded'}), 429
    
    try:
        body = request.get_json()
        validated = RecommendationRequestSchema(**body)
        
        # Initialisation des services
        embedding_service = EmbeddingService(supabase, redis_client)
        ab_testing = ABTestingFramework(supabase, redis_client)
        light_ml_model = LightMLModel(supabase)
        scoring_engine = ScoringEngine(embedding_service, light_ml_model)
        diversification_engine = DiversificationEngine()
        
        # Déterminer le groupe A/B
        ab_group = validated.context.ab_test_group if validated.context else None
        if not ab_group and validated.user_id:
            ab_group = ab_testing.assign_group(validated.user_id, 'recommendation_v3')
        else:
            ab_group = ab_group or 'control'
        
        weights_dict = WEIGHT_CONFIGS.get(ab_group, WEIGHT_CONFIGS['control'])
        weights = FeatureWeights(**weights_dict)
        
        # Cache key
        cache_key = f"rec:{validated.user_id or 'anon'}:{ab_group}:{validated.limit}:{validated.offset}"
        cached = global_cache.get(cache_key)
        if cached:
            cached['cached'] = True
            return jsonify(cached)
        
        # Récupération des données utilisateur
        user_profile = None
        user_favorites = []
        view_history = []
        
        if validated.user_id:
            # Profil
            profile_response = supabase.table('profiles')\
                .select('*')\
                .eq('user_id', validated.user_id)\
                .single()\
                .execute()
            
            if profile_response.data:
                data = profile_response.data
                user_profile = UserProfile(
                    user_id=data.get('user_id'),
                    city=data.get('city'),
                    budget_min=data.get('budget_min'),
                    budget_max=data.get('budget_max'),
                    preferred_property_types=data.get('preferred_property_types', []),
                    preferred_neighborhoods=data.get('preferred_neighborhoods', []),
                    preferred_listing_types=data.get('preferred_listing_types', []),
                    preferred_amenities=data.get('preferred_amenities', []),
                    move_in_timeline=data.get('move_in_timeline'),
                    user_type=data.get('user_type'),
                    segment=data.get('segment'),
                    furnished_preference=data.get('furnished_preference'),
                    parking_needed=data.get('parking_needed', False),
                    pet_friendly=data.get('pet_friendly', False),
                    must_have_features=data.get('must_have_features', []),
                    deal_breakers=data.get('deal_breakers', []),
                    preferred_floor=data.get('preferred_floor'),
                    kitchen_type_preference=data.get('kitchen_type_preference'),
                    min_bedrooms=data.get('min_bedrooms'),
                    min_bathrooms=data.get('min_bathrooms'),
                    needs_laundry=data.get('needs_laundry', False),
                    needs_dining_room=data.get('needs_dining_room', False)
                )
            
            # Favoris
            fav_response = supabase.table('property_favorites')\
                .select('property_id')\
                .eq('user_id', validated.user_id)\
                .execute()
            user_favorites = [f['property_id'] for f in (fav_response.data or [])]
            
            # Historique de vues
            view_response = supabase.table('property_views')\
                .select('property_id, view_duration_seconds, viewed_at')\
                .eq('user_id', validated.user_id)\
                .order('viewed_at', desc=True)\
                .limit(50)\
                .execute()
            view_history = view_response.data or []
        
        # Analyse des patterns et embeddings
        viewing_pattern = None
        viewed_property_ids = []
        final_user_embedding = None
        
        if view_history:
            engaged_views = [v for v in view_history if (v.get('view_duration_seconds') or 0) > 10]
            viewed_property_ids = [v['property_id'] for v in engaged_views]
            
            if viewed_property_ids:
                props_response = supabase.table('properties')\
                    .select('*, property_embeddings(vector)')\
                    .in_('id', viewed_property_ids[:30])\
                    .execute()
                
                if props_response.data:
                    viewing_pattern = analyze_viewing_pattern(props_response.data)
                    if validated.user_id:
                        final_user_embedding = await embedding_service.create_user_embedding(
                            validated.user_id, 
                            props_response.data
                        )
        
        if not final_user_embedding and validated.user_id:
            final_user_embedding = await embedding_service.get_user_embedding(validated.user_id)
        
        # Scores collaboratifs
        collaborative_scores = await get_collaborative_scores(supabase, validated.user_id, user_favorites)
        
        # Exclusions et candidats
        excluded_ids = list(set(user_favorites + viewed_property_ids))
        candidates = await generate_candidates(supabase, user_profile, validated.limit * 3, excluded_ids)
        
        if not candidates:
            return jsonify({
                'recommendations': [],
                'total': 0,
                'metadata': {
                    'ab_test_group': ab_group,
                    'processing_time_ms': int(time.time() * 1000 - request_start_time)
                }
            })
        
        # Scoring
        context_dict = validated.context.dict() if validated.context else None
        scored = [
            scoring_engine.calculate_property_score(
                p, user_profile, viewing_pattern, collaborative_scores,
                user_favorites, weights, context_dict, final_user_embedding
            )
            for p in candidates
        ]
        
        # Diversification
        diverse_results = diversification_engine.apply_mmr(scored, validated.limit, Config.MMR_LAMBDA)
        results = diverse_results[validated.offset:validated.offset + validated.limit]
        
        # Réponse
        response = {
            'recommendations': [
                {
                    **r.property,
                    '_score': r.score,
                    '_rank': r.rank,
                    '_reasons': r.reasons,
                    '_match_details': {
                        'collaborative_score': collaborative_scores.get(r.property['id'], 0),
                        'viewing_similarity': r.similarity_score,
                        'embedding_similarity': r.embedding_similarity,
                        'ml_score': r.ml_score,
                        'ab_test_group': ab_group,
                        'score_breakdown': r.score_breakdown
                    }
                }
                for r in results
            ],
            'total': len(diverse_results),
            'metadata': {
                'ab_test_group': ab_group,
                'processing_time_ms': int(time.time() * 1000 - request_start_time),
                'candidates_count': len(candidates),
                'user_embedding_used': final_user_embedding is not None,
                'profile_version': 'v2_full' if user_profile and user_profile.furnished_preference else 'v1',
                'property_types_considered': len(ALL_PROPERTY_TYPES),
                'cache_stats': global_cache.get_stats()
            }
        }
        
        # Mise en cache
        global_cache.set(cache_key, response, Config.CACHE_TTL)
        
        # Tracking A/B test
        if validated.user_id and results:
            await ab_testing.track_event({
                'user_id': validated.user_id,
                'experiment_id': 'recommendation_v3',
                'variant': ab_group,
                'event_type': 'impression',
                'property_id': results[0].property['id']
            })
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Recommendation error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/feedback', methods=['POST', 'OPTIONS'])
async def feedback():
    if request.method == 'OPTIONS':
        return '', 204
    
    try:
        body = request.get_json()
        validated = FeedbackEventSchema(**body)
        
        ab_testing = ABTestingFramework(supabase, redis_client)
        
        rewards = {
            'view': 0.1, 'click': 0.3, 'favorite': 0.5,
            'contact': 0.8, 'visit': 0.9, 'rent': 1.0
        }
        value = rewards.get(validated.event_type, 0)
        
        # Insertion parallèle
        supabase.table('feedback_events').insert({
            'user_id': validated.user_id,
            'property_id': validated.property_id,
            'event_type': validated.event_type,
            'value': value,
            'context': {
                **(validated.context or {}),
                'request_id': validated.request_id,
                'ab_test_group': validated.ab_test_group
            },
            'timestamp': datetime.utcnow().isoformat()
        }).execute()
        
        await ab_testing.track_event({
            'user_id': validated.user_id,
            'experiment_id': 'recommendation_v3',
            'variant': validated.ab_test_group or 'control',
            'event_type': validated.event_type,
            'property_id': validated.property_id,
            'value': value
        })
        
        return jsonify({'success': True, 'value': value})
        
    except Exception as e:
        logger.error(f"Feedback error: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.getenv('PORT', 10000))
    app.run(host='0.0.0.0', port=port)
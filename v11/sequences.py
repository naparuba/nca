import random
from typing import TYPE_CHECKING

from config import CONFIG
from simulator import get_simulator

if TYPE_CHECKING:
    from stages.base_stage import BaseStage


class OptimizedSequenceCache:
    """
    Cache spécialisé par étape pour l'entraînement modulaire.
    Maintient des caches séparés pour chaque étape d'apprentissage.
    """
    
    
    def __init__(self):
        self._stage_caches = {}  # Cache par étape
        self._current_indices = {}
    
    
    def initialize_stage_cache(self, stage):
        # type: (BaseStage) -> None
        """Initialise le cache pour une étape spécifique."""
        
        stage_nb = stage.get_stage_nb()
        if stage_nb in self._stage_caches:
            return  # Déjà initialisé
        
        cache_size = CONFIG.STAGE_CACHE_SIZE
        print(f"🎯 Génération de {cache_size} séquences pour l'étape {stage_nb}...", end='', flush=True)
        
        sequences = []
        for i in range(cache_size):
            if i % 50 == 0:
                print(f"\r   Étape {stage_nb}: {i}/{cache_size}                                 ", end='', flush=True)
            
            target_seq, source_mask, obstacle_mask = get_simulator().generate_stage_sequence(
                    stage=stage,
                    n_steps=CONFIG.NCA_STEPS,
                    size=CONFIG.GRID_SIZE
            )
            
            sequences.append({
                'target_seq':    target_seq,
                'source_mask':   source_mask,
                'obstacle_mask': obstacle_mask,
                'stage_nb':      stage_nb
            })
        
        self._stage_caches[stage_nb] = sequences
        self._current_indices[stage_nb] = 0
        print(f"\r✅ Cache étape {stage_nb} créé ({cache_size} séquences)")
    
    
    def get_stage_sample(self, stage):
        # type: (BaseStage) -> dict
        """Récupère un échantillon pour l'étape spécifiée."""
        stage_nb = stage.get_stage_nb()
        if stage_nb not in self._stage_caches:
            self.initialize_stage_cache(stage)
        
        cache = self._stage_caches[stage_nb]
        
        # Récupère l'échantillon courant et avance l'index
        sample = cache[self._current_indices[stage_nb]]
        self._current_indices[stage_nb] = (self._current_indices[stage_nb] + 1) % len(cache)
        
        return sample
    
    
    def shuffle_stage_cache(self, stage_nb):
        # type: (int) -> None
        """Mélange le cache d'une étape spécifique."""
        if stage_nb in self._stage_caches:
            random.shuffle(self._stage_caches[stage_nb])
    
    
    def clear_stage_cache(self, stage_nb):
        # type: (int) -> None
        """Libère la mémoire du cache d'une étape."""
        if stage_nb in self._stage_caches:
            del self._stage_caches[stage_nb]
            del self._current_indices[stage_nb]
            print(f"🗑️  Cache étape {stage_nb} libéré")

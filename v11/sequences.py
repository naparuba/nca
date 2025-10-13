from config import CONFIG
from simulator import get_simulator


class OptimizedSequenceCache:
    """
    Cache spécialisé par étape pour l'entraînement modulaire.
    Maintient des caches séparés pour chaque étape d'apprentissage.
    """
    
    
    def __init__(self):
        self.simulator = get_simulator()
        self.stage_caches = {}  # Cache par étape
        self.cache_sizes = {1: 150, 2: 200, 3: 250}  # Plus de variété pour étapes complexes
        self.current_indices = {}
    
    
    def initialize_stage_cache(self, stage_nb: int):
        """Initialise le cache pour une étape spécifique."""
        if stage_nb in self.stage_caches:
            return  # Déjà initialisé
        
        cache_size = self.cache_sizes.get(stage_nb, 200)
        print(f"🎯 Génération de {cache_size} séquences pour l'étape {stage_nb}...")
        
        sequences = []
        for i in range(cache_size):
            if i % 50 == 0:
                print(f"   Étape {stage_nb}: {i}/{cache_size}")
            
            target_seq, source_mask, obstacle_mask = self.simulator.generate_stage_sequence(
                    stage_nb=stage_nb,
                    n_steps=CONFIG.NCA_STEPS,
                    size=CONFIG.GRID_SIZE
            )
            
            sequences.append({
                'target_seq':    target_seq,
                'source_mask':   source_mask,
                'obstacle_mask': obstacle_mask,
                'stage_nb':      stage_nb
            })
        
        self.stage_caches[stage_nb] = sequences
        self.current_indices[stage_nb] = 0
        print(f"✅ Cache étape {stage_nb} créé ({cache_size} séquences)")
    
    
    def get_stage_batch(self, stage_nb: int, batch_size: int):
        """Récupère un batch pour l'étape spécifiée."""
        if stage_nb not in self.stage_caches:
            self.initialize_stage_cache(stage_nb)
        
        cache = self.stage_caches[stage_nb]
        batch = []
        
        for _ in range(batch_size):
            batch.append(cache[self.current_indices[stage_nb]])
            self.current_indices[stage_nb] = (self.current_indices[stage_nb] + 1) % len(cache)
        
        return batch
    
    
    def shuffle_stage_cache(self, stage_nb: int):
        """Mélange le cache d'une étape spécifique."""
        if stage_nb in self.stage_caches:
            import random
            random.shuffle(self.stage_caches[stage_nb])
    
    
    def clear_stage_cache(self, stage_nb: int):
        """Libère la mémoire du cache d'une étape."""
        if stage_nb in self.stage_caches:
            del self.stage_caches[stage_nb]
            del self.current_indices[stage_nb]
            print(f"🗑️  Cache étape {stage_nb} libéré")

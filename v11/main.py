import os
# HACK for imports
import sys
import time

import numpy as np
import torch

from nca_model import NCA
from stage_manager import STAGE_MANAGER
from trainer import Trainer
from visualizer import ProgressiveVisualizer

# Hack for imports
sys.path.append(os.path.dirname(__file__))

from config import CONFIG, DEVICE

# Initialisation
torch.manual_seed(CONFIG.SEED)
np.random.seed(CONFIG.SEED)

# Création du répertoire de sortie avec seed
CONFIG.OUTPUT_DIR = f"outputs"
os.makedirs(CONFIG.OUTPUT_DIR, exist_ok=True)

print(f"Device: {DEVICE}")
print(f"Seed: {CONFIG.SEED}")
print(f"Répertoire de sortie: {CONFIG.OUTPUT_DIR}")


def main():
    """
    Fonction principale pour l'entraînement modulaire progressif.
    Orchestre tout le processus d'apprentissage par curriculum.
    """
    print(f"\n" + "=" * 80)
    print(f"🚀 NEURAL CELLULAR AUTOMATON - APPRENTISSAGE MODULAIRE")
    print(f"=" * 80)
    
    try:
        # Initialisation du modèle
        print("\n🔧 Initialisation du modèle...")
        model = NCA(input_size=11,  # 9 (patch 3x3) + 1 (source) + 1 (obstacle)
                    ).to(DEVICE)
        
        print(f"📊 Nombre de paramètres dans le modèle: {sum(p.numel() for p in model.parameters()):,}")
        
        # Initialisation de l'entraîneur modulaire
        print("🎯 Initialisation de l'entraîneur modulaire...")
        trainer = Trainer(model)
        
        # Lancement de l'entraînement complet
        print("🚀 Lancement de l'entraînement modulaire...")
        before = time.time()
        trainer.train_full_curriculum()
        after = time.time()
        
        # Génération des visualisations progressives
        print("\n🎨 Génération des visualisations...")
        
        visualizer = ProgressiveVisualizer()
        
        # Visualisation par étape avec le modèle final
        for stage in STAGE_MANAGER.get_stages():
            visualizer.visualize_stage_results(model, stage)
        
        # Résumé visuel complet du curriculum
        visualizer.create_curriculum_summary()
        
        # Rapport final
        print(f"\n" + "=" * 80)
        print(f"🎉 ENTRAÎNEMENT MODULAIRE TERMINÉ AVEC SUCCÈS!")
        print(f"=" * 80)
        print(f"📁 Résultats sauvegardés dans: {CONFIG.OUTPUT_DIR}")
        print(f"⏱️  Temps total: {f"{(after - before) / 60:.1f} min"}")
        print(f"📊 Époques: {CONFIG.TOTAL_EPOCHS}")
    
    except KeyboardInterrupt:
        print(f"\n⚠️  Entraînement interrompu par l'utilisateur")
        print(f"\n❌ Programme terminé avec erreurs")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ ERREUR lors de l'entraînement:")
        print(f"   {type(e).__name__}: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    # Exécution du programme principal
    results = main()
    print(f"\n🎯 Programme terminé avec succès!")

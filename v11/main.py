import os
# HACK for imports
import sys

import numpy as np
import torch

from nca_model import ImprovedNCA
from trainer import ModularTrainer
from visualizer import ProgressiveVisualizer

# Hack for imports
sys.path.append(os.path.dirname(__file__))

from config import CONFIG
from torching import DEVICE

# =============================================================================
# Configuration et initialisation modulaire
# =============================================================================


# Initialisation
torch.manual_seed(CONFIG.SEED)
np.random.seed(CONFIG.SEED)

# Création du répertoire de sortie avec seed
CONFIG.OUTPUT_DIR = f"outputs"
os.makedirs(CONFIG.OUTPUT_DIR, exist_ok=True)

print(f"Device: {DEVICE}")
print(f"Seed: {CONFIG.SEED}")
print(f"Répertoire de sortie: {CONFIG.OUTPUT_DIR}")

from stage_manager import STAGE_MANAGER


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
        model = ImprovedNCA(input_size=11,  # 9 (patch 3x3) + 1 (source) + 1 (obstacle)
                            ).to(DEVICE)
        
        print(f"📊 Nombre de paramètres dans le modèle: {sum(p.numel() for p in model.parameters()):,}")
        
        # Initialisation de l'entraîneur modulaire
        print("🎯 Initialisation de l'entraîneur modulaire...")
        trainer = ModularTrainer(model)
        
        # Lancement de l'entraînement complet
        print("🚀 Lancement de l'entraînement modulaire...")
        global_metrics = trainer.train_full_curriculum()
        
        # Génération des visualisations progressives
        print("\n🎨 Génération des visualisations...")
        
        visualizer = ProgressiveVisualizer()
        
        # Visualisation par étape avec le modèle final
        for stage_nb in [1, 2, 3]:
            visualizer.visualize_stage_results(model, stage_nb)
        
        # Résumé visuel complet du curriculum
        visualizer.create_curriculum_summary(global_metrics)
        
        # Rapport final
        print(f"\n" + "=" * 80)
        print(f"🎉 ENTRAÎNEMENT MODULAIRE TERMINÉ AVEC SUCCÈS!")
        print(f"=" * 80)
        print(f"📁 Résultats sauvegardés dans: {CONFIG.OUTPUT_DIR}")
        print(f"⏱️  Temps total: {global_metrics['total_time_formatted']}")
        print(f"📊 Époques: {global_metrics['total_epochs_actual']}/{global_metrics['total_epochs_planned']}")
        print(f"📉 Perte finale: {global_metrics['final_loss']:.6f}")
        
        # Détail par étape
        print(f"\n📋 DÉTAIL PAR ÉTAPE:")
        for stage in STAGE_MANAGER.get_stages():
            stage_data = global_metrics['stage_metrics'][stage.get_stage_nb()]
            print(f"   Étape {stage_nb} ({stage.get_display_name()}): {stage_data['final_loss']:.6f}")
        
        print(f"\n🎨 Fichiers de visualisation générés:")
        print(f"   • Animations par étape: stage_X/")
        print(f"   • Progression curriculum: curriculum_progression.png")
        print(f"   • Comparaison étapes: stage_comparison.png")
        print(f"   • Résumé performance: performance_summary.png")
        print(f"   • Métriques complètes: complete_metrics.json")
        
        return global_metrics
    
    except KeyboardInterrupt:
        print(f"\n⚠️  Entraînement interrompu par l'utilisateur")
        return None
    except Exception as e:
        print(f"\n❌ ERREUR lors de l'entraînement:")
        print(f"   {type(e).__name__}: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    # Exécution du programme principal
    results = main()
    
    if results is not None:
        print(f"\n🎯 Programme terminé avec succès!")
        print(f"📊 Résultats disponibles dans la variable 'results'")
    else:
        print(f"\n❌ Programme terminé avec erreurs")
        exit(1)

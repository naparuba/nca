import argparse
import os
# HACK for imports
import sys
import time
import traceback

import numpy as np
import torch

from nca_model import NCA
from stage_manager import STAGE_MANAGER
from trainer import Trainer
from visualizer import get_visualizer

# Hack for imports
sys.path.append(os.path.dirname(__file__))

from config import CONFIG, DEVICE


def parse_command_line_args() -> None:
    """
    Parse les arguments de ligne de commande et met à jour la configuration.
    
    Permet de surcharger les valeurs par défaut de CONFIG via des arguments CLI.
    Cette approche facilite l'expérimentation sans modifier le code source.
    
    Arguments supportés:
    - --epochs: Nombre d'époques par étape d'entraînement
    - --nca-steps: Nombre d'étapes NCA par séquence de simulation
    - --hidden-size: Taille de la couche cachée du réseau de neurones
    - --n-layers: Nombre de couches du réseau de neurones
    """
    parser = argparse.ArgumentParser(description="Neural Cellular Automaton - Apprentissage modulaire progressif")
    # Arguments pour surcharger les paramètres d'entraînement
    parser.add_argument("--epochs", type=int, default=CONFIG.NB_EPOCHS_BY_STAGE, help="Epoques par stage")
    parser.add_argument("--nca-steps", type=int, default=CONFIG.NCA_STEPS, help="étapes temporelle par simulation")
    # Arguments pour surcharger l'architecture du modèle
    parser.add_argument("--hidden-size", type=int, default=CONFIG.HIDDEN_SIZE)
    parser.add_argument("--n-layers", type=int, default=CONFIG.N_LAYERS)
    
    # Parse des arguments et mise à jour de la configuration
    args = parser.parse_args()
    
    # Application des valeurs surchargées à la configuration globale
    # Ces modifications affectent tous les composants qui utilisent CONFIG
    CONFIG.NB_EPOCHS_BY_STAGE = args.epochs
    CONFIG.NCA_STEPS = args.nca_steps
    CONFIG.HIDDEN_SIZE = args.hidden_size
    CONFIG.N_LAYERS = args.n_layers
    
    # Recalcul du nombre total d'époques basé sur la nouvelle valeur par étape
    CONFIG.TOTAL_EPOCHS = CONFIG.NB_EPOCHS_BY_STAGE * len(STAGE_MANAGER.get_stages())
    
    # Affichage des paramètres effectifs pour confirmation
    print(f"📋 Configuration effective:")
    print(f"   • Époques par étape: {CONFIG.NB_EPOCHS_BY_STAGE}")
    print(f"   • Étapes NCA: {CONFIG.NCA_STEPS}")
    print(f"   • Taille cachée: {CONFIG.HIDDEN_SIZE}")
    print(f"   • Nombre de couches: {CONFIG.N_LAYERS}")
    print(f"   • Total époques: {CONFIG.TOTAL_EPOCHS}")


# Initialisation - Application des arguments CLI avant tout le reste
parse_command_line_args()

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
        model = NCA().to(DEVICE)
        
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
        
        visualizer = get_visualizer()
        
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
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    # Exécution du programme principal
    results = main()
    print(f"\n🎯 Programme terminé avec succès!")

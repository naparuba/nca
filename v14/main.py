import argparse
import os
# HACK for imports
import sys
import time
import traceback

from nca_model import NCA
from stage_manager import STAGE_MANAGER
from torched import set_random_seed
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
    - --skip-if-already: Évite de ré-entraîner une configuration déjà évaluée
    - --visualization-only: Génère uniquement les graphiques de performance sans entraînement
    """
    parser = argparse.ArgumentParser(description="Neural Cellular Automaton - Apprentissage modulaire progressif")
    # Arguments pour surcharger les paramètres d'entraînement
    parser.add_argument("--epochs", type=int, default=CONFIG.NB_EPOCHS_BY_STAGE, help="Epoques par stage")
    parser.add_argument("--nca-steps", type=int, default=CONFIG.NCA_STEPS, help="étapes temporelle par simulation")
    # Arguments pour surcharger l'architecture du modèle
    parser.add_argument("--hidden-size", type=int, default=CONFIG.HIDDEN_SIZE)
    parser.add_argument("--n-layers", type=int, default=CONFIG.N_LAYERS)
    # Optimisation pour éviter les calculs redondants
    parser.add_argument("--skip-if-already", action="store_true",
                        help="Saute l'entraînement si la configuration a déjà été évaluée")
    # Mode visualisation uniquement
    parser.add_argument("--visualization-only", action="store_true",
                        help="Génère uniquement les graphiques de performance sans entraînement")
    
    # Parse des arguments et mise à jour de la configuration
    args = parser.parse_args()
    
    # Application des valeurs surchargées à la configuration globale
    # Ces modifications affectent tous les composants qui utilisent CONFIG
    CONFIG.NB_EPOCHS_BY_STAGE = args.epochs
    CONFIG.NCA_STEPS = args.nca_steps
    CONFIG.HIDDEN_SIZE = args.hidden_size
    CONFIG.N_LAYERS = args.n_layers
    CONFIG.SKIP_IF_ALREADY = args.skip_if_already
    CONFIG.VISUALIZATION_ONLY = args.visualization_only
    
    # Recalcul du nombre total d'époques basé sur la nouvelle valeur par étape
    CONFIG.TOTAL_EPOCHS = CONFIG.NB_EPOCHS_BY_STAGE * len(STAGE_MANAGER.get_stages())
    
    # Affichage des paramètres effectifs pour confirmation
    print(f"📋 Configuration effective:")
    print(f"   • Époques par étape: {CONFIG.NB_EPOCHS_BY_STAGE}")
    print(f"   • Étapes NCA: {CONFIG.NCA_STEPS}")
    print(f"   • Taille cachée: {CONFIG.HIDDEN_SIZE}")
    print(f"   • Nombre de couches: {CONFIG.N_LAYERS}")
    print(f"   • Total époques: {CONFIG.TOTAL_EPOCHS}")
    print(f"   • Skip si déjà évalué: {CONFIG.SKIP_IF_ALREADY}")
    print(f"   • Mode visualisation uniquement: {CONFIG.VISUALIZATION_ONLY}")


# Initialisation - Application des arguments CLI avant tout le reste
parse_command_line_args()

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
    
    set_random_seed(CONFIG.SEED)  # Assure la reproductibilité
    
    try:
        # Mode visualisation uniquement : génère les graphiques sans entraînement
        if CONFIG.VISUALIZATION_ONLY:
            print("\n📊 Mode visualisation uniquement activé")
            print("🎨 Génération des graphiques de performance...")
            
            visualizer = get_visualizer()
            visualizer.plot_performance_comparison()
            
            print(f"\n" + "=" * 80)
            print(f"✅ VISUALISATIONS GÉNÉRÉES AVEC SUCCÈS!")
            print(f"=" * 80)
            print(f"📁 Graphiques sauvegardés dans: {CONFIG.OUTPUT_DIR}")
            return
        
        # Vérification si la configuration a déjà été évaluée (si option activée)
        if CONFIG.SKIP_IF_ALREADY:
            print("\n🔍 Vérification si la configuration a déjà été évaluée...")
            visualizer = get_visualizer()
            
            # Vérifier pour tous les stages si la configuration existe
            all_stages_already_evaluated = True
            for stage in STAGE_MANAGER.get_stages():
                stage_nb = stage.get_stage_nb()
                already_evaluated = visualizer.check_configuration_already_evaluated(
                        stage_nb=stage_nb,
                        n_layers=CONFIG.N_LAYERS,
                        hidden_size=CONFIG.HIDDEN_SIZE,
                        nb_epochs_trained=CONFIG.NB_EPOCHS_BY_STAGE
                )
                
                if not already_evaluated:
                    all_stages_already_evaluated = False
                    print(f"   ❌ Stage {stage_nb}: Non évalué")
                else:
                    print(f"   ✅ Stage {stage_nb}: Déjà évalué")
            
            if all_stages_already_evaluated:
                print(
                    f"\n⏭️  Configuration déjà évaluée pour tous les stages! N_LAYERS={CONFIG.N_LAYERS} HIDDEN_SIZE={CONFIG.HIDDEN_SIZE} NB_EPOCHS={CONFIG.NB_EPOCHS_BY_STAGE}")
                sys.exit(0)
        
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
        visualizer.plot_performance_comparison()
        
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

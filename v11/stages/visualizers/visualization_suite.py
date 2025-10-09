"""
Suite complète de visualisation pour NCA modulaire avec intensités variables.
Migré depuis visualize_modular_progressive_obstacles_variable_intensity.py
"""

from typing import Dict, Any
from pathlib import Path

from .progressive_visualizer import ProgressiveVisualizer


def create_complete_visualization_suite(model, global_metrics: Dict[str, Any],
                                      simulator, cfg):
    """
    Crée la suite complète de visualisations pour la version 8__.
    
    Comprend:
    - Visualisations par étape avec intensités
    - Comparaisons d'intensités pour le stage d'intensité variable
    - Graphiques de curriculum étendu
    - Métriques spécialisées intensités variables
    
    Args:
        model: Modèle NCA entraîné
        global_metrics: Métriques globales d'entraînement
        simulator: Simulateur de diffusion
        cfg: Configuration du système
    """
    print("\n" + "="*80)
    print("🎨 GÉNÉRATION DE LA SUITE COMPLÈTE DE VISUALISATIONS v8__")
    print("="*80)
    
    # Initialisation du visualiseur progressif
    visualizer = ProgressiveVisualizer()
    
    # 1. Visualisations par étape (étendues)
    print("\n🎨 Génération des visualisations par étape...")
    
    # Récupération de la séquence de stages depuis le simulateur
    from ..sequence import StageSequence
    sequence = StageSequence()
    stage_names = sequence.get_sequence()
    
    # Visualisations pour tous les stages sauf le dernier (intensité variable)
    for stage_name in stage_names[:-1]:  # Exclut 'variable_intensity'
        print(f"🎨 Génération des visualisations pour le stage '{stage_name}'...")
        stage_vis = visualizer.visualize_stage_results(model, stage_name, simulator, cfg)

    # Stage d'intensité variable: plusieurs intensités pour analyse comparative
    variable_intensity_stage = 'variable_intensity'
    if variable_intensity_stage in stage_names:
        print(f"🎨 Génération des visualisations pour le stage '{variable_intensity_stage}' avec intensités multiples...")
        variable_intensity_intensities = [0.0, 0.3, 0.7, 1.0]
        for intensity in variable_intensity_intensities:
            stage_vis = visualizer.visualize_stage_results(
                model, stage=variable_intensity_stage, simulator=simulator, cfg=cfg,
                source_intensity=intensity)

    
    # 2. Grille comparative d'intensités
    print("\n🎨 Génération de la grille comparative d'intensités...")
    visualizer.create_intensity_comparison_grid(model, simulator, cfg)

    
    # 3. Curriculum d'intensité (nouveau)
    print("\n🎨 Génération des graphiques de curriculum d'intensité...")
    variable_intensity_metrics_key = f"{variable_intensity_stage}_metrics"
    if variable_intensity_metrics_key in global_metrics:
        visualizer.visualize_intensity_curriculum(global_metrics[variable_intensity_metrics_key], cfg)
    else:
        print(f"⚠️ Métriques du stage '{variable_intensity_stage}' non disponibles pour le curriculum d'intensité")

    
    # 4. Résumé visuel étendu
    print("\n🎨 Génération du résumé visuel complet étendu...")
    visualizer.create_curriculum_summary_extended(global_metrics, cfg)

    
    print("\n" + "="*80)
    print("✅ SUITE COMPLÈTE DE VISUALISATIONS v8__ GÉNÉRÉE!")
    print("="*80)
    
    # Résumé des fichiers générés pour information
    output_dir = Path(cfg.OUTPUT_DIR)
    print(f"\n📁 Fichiers générés dans: {output_dir}")
    print("📋 Liste des visualisations:")
    print("   • Animations par étape: stage_X/")
    print("   • Grille comparative intensités: intensity_comparison_grid.png")
    print("   • Distribution intensités: intensity_distribution.png")
    print("   • Performance par plage: performance_by_intensity_range.png")
    print("   • Convergence vs intensité: convergence_analysis_by_intensity.png")
    print("   • Progression curriculum étendu: curriculum_progression_extended.png")
    print("   • Comparaison étapes étendue: stage_comparison_extended.png")
    print("   • Résumé performance étendu: performance_summary_extended.png")

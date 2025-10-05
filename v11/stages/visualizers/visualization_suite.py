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
    - Visualisations par étape (1-4) avec intensités
    - Comparaisons d'intensités pour l'étape 4
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
    
    # Étapes 1-3: intensité standard
    for stage in [1, 2, 3]:
        stage_vis = visualizer.visualize_stage_results(model, stage, simulator, cfg)

    # Étape 4: plusieurs intensités pour analyse comparative
    stage_4_intensities = [0.0, 0.3, 0.7, 1.0]
    for intensity in stage_4_intensities:
        stage_vis = visualizer.visualize_stage_results(model, stage=4, simulator=simulator, cfg=cfg,
                                                         source_intensity=intensity)

    
    # 2. Grille comparative d'intensités
    print("\n🎨 Génération de la grille comparative d'intensités...")
    visualizer.create_intensity_comparison_grid(model, simulator, cfg)

    
    # 3. Curriculum d'intensité (nouveau)
    print("\n🎨 Génération des graphiques de curriculum d'intensité...")
    if 'stage_4_metrics' in global_metrics:
        visualizer.visualize_intensity_curriculum(global_metrics['stage_4_metrics'], cfg)
    else:
        print("⚠️ Métriques étape 4 non disponibles pour le curriculum d'intensité")

    
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

"""
Générateur de graphiques spécialisés pour les métriques d'intensité variable.
Migré depuis visualize_modular_progressive_obstacles_variable_intensity.py
"""

import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict, Any
from pathlib import Path


class VariableIntensityMetricsPlotter:
    """
    Générateur de graphiques spécialisés pour les métriques d'intensité variable.
    """
    
    def plot_intensity_distribution(self, intensity_history: List[float],
                                  output_dir: Path):
        """
        Graphique de distribution des intensités utilisées pendant l'entraînement.
        
        Args:
            intensity_history: Historique des intensités utilisées
            output_dir: Répertoire de sortie pour sauvegarder le graphique
        """
        if not intensity_history:
            print("⚠️ Pas d'historique d'intensité disponible")
            return
            
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Histogramme de distribution
        ax1.hist(intensity_history, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        ax1.set_xlabel('Intensité de Source')
        ax1.set_ylabel('Fréquence')
        ax1.set_title('Distribution des Intensités (Étape 4)')
        ax1.grid(True, alpha=0.3)
        
        # Ajout de statistiques descriptives
        mean_intensity = np.mean(intensity_history)
        std_intensity = np.std(intensity_history)
        ax1.axvline(mean_intensity, color='red', linestyle='--',
                   label=f'Moyenne: {mean_intensity:.3f}')
        ax1.legend()
        
        # Évolution temporelle des intensités
        ax2.plot(intensity_history, 'o-', alpha=0.6, markersize=1)
        ax2.set_xlabel('Simulation #')
        ax2.set_ylabel('Intensité')
        ax2.set_title('Évolution des Intensités au Cours de l\'Entraînement')
        ax2.grid(True, alpha=0.3)
        
        # Ligne de tendance pour voir l'évolution
        if len(intensity_history) > 1:
            z = np.polyfit(range(len(intensity_history)), intensity_history, 1)
            p = np.poly1d(z)
            ax2.plot(range(len(intensity_history)), p(range(len(intensity_history))),
                    "r--", alpha=0.8, label=f'Tendance: {z[0]:.6f}x + {z[1]:.3f}')
            ax2.legend()
        
        plt.tight_layout()
        plt.savefig(output_dir / "intensity_distribution.png", dpi=150, bbox_inches='tight')
        plt.close()
        
        print("✅ Graphique de distribution des intensités généré")
    
    def plot_performance_by_intensity_range(self, metrics_by_intensity: Dict[str, List[float]],
                                          output_dir: Path):
        """
        Performance du modèle selon les plages d'intensité.
        
        Args:
            metrics_by_intensity: Dictionnaire avec intensités et pertes correspondantes
            output_dir: Répertoire de sortie
        """
        if 'intensities' not in metrics_by_intensity or 'losses' not in metrics_by_intensity:
            print("⚠️ Données de performance par intensité manquantes")
            return
            
        # Regroupement par plages d'intensité pour analyse comparative
        ranges = {
            'Très faible\n(0.0-0.2)': [],
            'Faible\n(0.2-0.4)': [],
            'Moyenne\n(0.4-0.6)': [],
            'Forte\n(0.6-0.8)': [],
            'Très forte\n(0.8-1.0)': []
        }
        
        # Classification des performances par plage
        for intensity, loss in zip(metrics_by_intensity['intensities'],
                                 metrics_by_intensity['losses']):
            if intensity <= 0.2:
                ranges['Très faible\n(0.0-0.2)'].append(loss)
            elif intensity <= 0.4:
                ranges['Faible\n(0.2-0.4)'].append(loss)
            elif intensity <= 0.6:
                ranges['Moyenne\n(0.4-0.6)'].append(loss)
            elif intensity <= 0.8:
                ranges['Forte\n(0.6-0.8)'].append(loss)
            else:
                ranges['Très forte\n(0.8-1.0)'].append(loss)
        
        # Graphique en boîtes pour comparer les performances
        fig, ax = plt.subplots(figsize=(12, 8))
        
        data_to_plot = [losses for losses in ranges.values() if losses]
        labels = [label for label, losses in ranges.items() if losses]
        
        if data_to_plot:
            box_plot = ax.boxplot(data_to_plot, labels=labels, patch_artist=True)
            
            # Coloration des boîtes pour une meilleure lisibilité
            colors = ['lightcoral', 'lightsalmon', 'lightblue', 'lightgreen', 'lightsteelblue']
            for patch, color in zip(box_plot['boxes'], colors[:len(box_plot['boxes'])]):
                patch.set_facecolor(color)
            
            ax.set_ylabel('Perte MSE')
            ax.set_title('Performance par Plage d\'Intensité (Étape 4)')
            ax.grid(True, alpha=0.3)
            ax.set_yscale('log')
            
            plt.tight_layout()
            plt.savefig(output_dir / "performance_by_intensity_range.png",
                       dpi=150, bbox_inches='tight')
            plt.close()
            
            print("✅ Graphique de performance par plage d'intensité généré")
        else:
            print("⚠️ Pas assez de données pour les plages d'intensité")
    
    def plot_convergence_analysis_by_intensity(self, convergence_data: Dict[str, Any],
                                             output_dir: Path):
        """
        Analyse de convergence selon l'intensité.
        
        Args:
            convergence_data: Données de convergence avec intensités, temps et scores
            output_dir: Répertoire de sortie
        """
        required_keys = ['intensities', 'convergence_times', 'stability_scores']
        if not all(key in convergence_data for key in required_keys):
            print("⚠️ Données de convergence par intensité incomplètes")
            return
            
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Analyse du temps de convergence vs intensité
        intensities = convergence_data['intensities']
        convergence_times = convergence_data['convergence_times']
        
        scatter = ax1.scatter(intensities, convergence_times, alpha=0.6, c=intensities,
                            cmap='viridis', s=50)
        ax1.set_xlabel('Intensité de Source')
        ax1.set_ylabel('Temps de Convergence (époques)')
        ax1.set_title('Temps de Convergence vs Intensité')
        ax1.grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=ax1)
        
        # Analyse de la stabilité vs intensité
        stability_scores = convergence_data['stability_scores']
        
        scatter2 = ax2.scatter(intensities, stability_scores, alpha=0.6, c=intensities,
                             cmap='plasma', s=50)
        ax2.set_xlabel('Intensité de Source')
        ax2.set_ylabel('Score de Stabilité')
        ax2.set_title('Stabilité vs Intensité')
        ax2.grid(True, alpha=0.3)
        plt.colorbar(scatter2, ax=ax2)
        
        plt.tight_layout()
        plt.savefig(output_dir / "convergence_analysis_by_intensity.png",
                   dpi=150, bbox_inches='tight')
        plt.close()
        
        print("✅ Graphique d'analyse de convergence par intensité généré")

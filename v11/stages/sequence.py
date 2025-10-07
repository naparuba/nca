"""
Définition de la séquence d'exécution des stages.
C'EST LE SEUL ENDROIT où l'ordre des stages est défini.

Principe:
- Une liste maître définit l'ordre d'exécution des stages
- Les stages eux-mêmes ne connaissent PAS leur position
- Permet l'insertion, suppression et réorganisation facile
- Source unique de vérité pour l'ordonnancement

Avantages:
- Flexibilité totale: modifier l'ordre sans toucher au code des stages
- Insertion facile: ajouter un stage entre deux autres
- Maintenance simple: un seul fichier à modifier
"""

from typing import List, Optional


class StageSequence:
    """
    Définit l'ordre d'exécution des stages.
    
    La séquence est définie par une liste de slugs (identifiants uniques).
    Les stages sont exécutés dans l'ordre de cette liste.
    
    Cette classe est la SEULE source de vérité pour l'ordre d'exécution.
    Les stages eux-mêmes n'ont aucune connaissance de leur position.
    """
    
    # 🎯 SÉQUENCE MAÎTRE - L'ORDRE EST DÉFINI ICI ET NULLE PART AILLEURS
    # Pour modifier l'ordre d'exécution, il suffit de réorganiser cette liste
    # Pour ajouter un stage, il suffit de l'insérer à la position voulue
    DEFAULT_SEQUENCE = [
        'no_obstacles',           # Apprentissage de base sans obstacles
        'single_obstacle',        # Introduction d'un obstacle unique
        'multiple_obstacles',     # Gestion de plusieurs obstacles
        'variable_intensity',     # Intensité variable de la source
        'time_attenuation',       # Atténuation temporelle
    ]
    
    def __init__(self, custom_sequence: Optional[List[str]] = None):
        """
        Initialise la séquence.
        
        Args:
            custom_sequence: Séquence personnalisée (optionnel)
                           Si None, utilise DEFAULT_SEQUENCE
        """
        # Copie de la séquence pour éviter les modifications accidentelles
        self.sequence = custom_sequence.copy() if custom_sequence else self.DEFAULT_SEQUENCE.copy()
    
    def get_sequence(self) -> List[str]:
        """
        Retourne la séquence ordonnée des stages.
        
        Returns:
            Liste des slugs dans l'ordre d'exécution
        """
        return self.sequence.copy()  # Copie pour immutabilité
    
    def get_position(self, slug: str) -> int:
        """
        Retourne la position d'un stage dans la séquence (0-indexed).
        
        Args:
            slug: Identifiant du stage
            
        Returns:
            Index du stage dans la séquence
            
        Raises:
            ValueError: Si le stage n'est pas dans la séquence
        """
        try:
            return self.sequence.index(slug)
        except ValueError:
            raise ValueError(
                f"Stage '{slug}' non trouvé dans la séquence.\n"
                f"Séquence actuelle: {', '.join(self.sequence)}"
            )
    
    def insert_before(self, new_slug: str, reference_slug: str):
        """
        Insère un nouveau stage avant un stage de référence.
        
        Exemple:
            sequence.insert_before('new_stage', 'single_obstacle')
            # Résultat: [..., 'no_obstacles', 'new_stage', 'single_obstacle', ...]
        
        Args:
            new_slug: Slug du stage à insérer
            reference_slug: Slug du stage de référence
            
        Raises:
            ValueError: Si le stage de référence n'existe pas
        """
        position = self.get_position(reference_slug)
        self.sequence.insert(position, new_slug)
        print(f"✅ Stage '{new_slug}' inséré avant '{reference_slug}'")
    
    def insert_after(self, new_slug: str, reference_slug: str):
        """
        Insère un nouveau stage après un stage de référence.
        
        Exemple:
            sequence.insert_after('new_stage', 'no_obstacles')
            # Résultat: [..., 'no_obstacles', 'new_stage', 'single_obstacle', ...]
        
        Args:
            new_slug: Slug du stage à insérer
            reference_slug: Slug du stage de référence
            
        Raises:
            ValueError: Si le stage de référence n'existe pas
        """
        position = self.get_position(reference_slug)
        self.sequence.insert(position + 1, new_slug)
        print(f"✅ Stage '{new_slug}' inséré après '{reference_slug}'")
    
    def remove(self, slug: str):
        """
        Retire un stage de la séquence.
        
        Args:
            slug: Identifiant du stage à retirer
            
        Raises:
            ValueError: Si le stage n'existe pas dans la séquence
        """
        try:
            self.sequence.remove(slug)
            print(f"✅ Stage '{slug}' retiré de la séquence")
        except ValueError:
            raise ValueError(f"Stage '{slug}' non trouvé dans la séquence")
    
    def reorder(self, new_sequence: List[str]):
        """
        Remplace complètement la séquence par une nouvelle.
        
        Utile pour une réorganisation complète de l'ordre d'exécution.
        
        Args:
            new_sequence: Nouvelle liste de slugs ordonnés
        """
        self.sequence = new_sequence.copy()
        print(f"✅ Séquence réorganisée: {', '.join(self.sequence)}")
    
    def append(self, slug: str):
        """
        Ajoute un stage à la fin de la séquence.
        
        Args:
            slug: Identifiant du stage à ajouter
        """
        self.sequence.append(slug)
        print(f"✅ Stage '{slug}' ajouté à la fin de la séquence")
    
    def prepend(self, slug: str):
        """
        Ajoute un stage au début de la séquence.
        
        Args:
            slug: Identifiant du stage à ajouter
        """
        self.sequence.insert(0, slug)
        print(f"✅ Stage '{slug}' ajouté au début de la séquence")
    
    def get_length(self) -> int:
        """
        Retourne le nombre de stages dans la séquence.
        
        Returns:
            Nombre de stages
        """
        return len(self.sequence)
    
    def contains(self, slug: str) -> bool:
        """
        Vérifie si un stage est dans la séquence.
        
        Args:
            slug: Identifiant du stage
            
        Returns:
            True si le stage est dans la séquence
        """
        return slug in self.sequence
    
    def get_next_stage(self, current_slug: str) -> Optional[str]:
        """
        Retourne le slug du stage suivant dans la séquence.
        
        Args:
            current_slug: Slug du stage actuel
            
        Returns:
            Slug du stage suivant ou None si c'est le dernier
            
        Raises:
            ValueError: Si le stage actuel n'est pas dans la séquence
        """
        position = self.get_position(current_slug)
        
        if position < len(self.sequence) - 1:
            return self.sequence[position + 1]
        
        return None
    
    def get_previous_stage(self, current_slug: str) -> Optional[str]:
        """
        Retourne le slug du stage précédent dans la séquence.
        
        Args:
            current_slug: Slug du stage actuel
            
        Returns:
            Slug du stage précédent ou None si c'est le premier
            
        Raises:
            ValueError: Si le stage actuel n'est pas dans la séquence
        """
        position = self.get_position(current_slug)
        
        if position > 0:
            return self.sequence[position - 1]
        
        return None
    
    def is_first(self, slug: str) -> bool:
        """
        Vérifie si un stage est le premier de la séquence.
        
        Args:
            slug: Identifiant du stage
            
        Returns:
            True si le stage est en première position
        """
        return self.sequence[0] == slug if self.sequence else False
    
    def is_last(self, slug: str) -> bool:
        """
        Vérifie si un stage est le dernier de la séquence.
        
        Args:
            slug: Identifiant du stage
            
        Returns:
            True si le stage est en dernière position
        """
        return self.sequence[-1] == slug if self.sequence else False
    
    def __repr__(self) -> str:
        """Représentation textuelle de la séquence."""
        return f"StageSequence({' → '.join(self.sequence)})"
    
    def __len__(self) -> int:
        """Permet d'utiliser len() sur la séquence."""
        return len(self.sequence)
    
    def __iter__(self):
        """Permet d'itérer directement sur la séquence."""
        return iter(self.sequence)


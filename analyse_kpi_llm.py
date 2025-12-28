"""
Script d'analyse complémentaire des KPIs utilisant un LLM alternatif (Gemini)
Ce script lit les données JSON générées par le code principal et produit
une analyse détaillée des KPIs avec une conclusion générale.
"""

import json
import os
import google.generativeai as genai
from datetime import datetime
import streamlit as st

# Configuration Gemini
GEMINI_CONFIG = {
    "api_key": os.getenv("GEMINI_API_KEY", "AIzaSyAFYfD6CqBt1WO7w4b6Xn8RTe5tiZSKNKg")
}

# Initialisation Gemini
genai.configure(api_key=GEMINI_CONFIG["api_key"])


class KPIAnalyzerLLM:
    """Analyseur de KPIs utilisant Gemini pour une perspective complémentaire"""
    
    def __init__(self):
        try:
            self.model = genai.GenerativeModel('gemini-pro')
        except Exception as e:
            st.error(f"Erreur initialisation Gemini: {str(e)}")
            self.model = None
    
    def load_json_data(self, json_path=None, json_data=None):
        """
        Charge les données JSON depuis un fichier ou directement depuis un objet
        
        Args:
            json_path: Chemin vers le fichier JSON (optionnel)
            json_data: Données JSON directement (optionnel)
        
        Returns:
            dict: Données JSON chargées
        """
        if json_data:
            return json_data
        elif json_path and os.path.exists(json_path):
            with open(json_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        else:
            raise ValueError("Aucune donnée JSON fournie (ni json_path ni json_data)")
    
    def prepare_kpi_analysis_prompt(self, analysis_json):
        """
        Prépare le prompt pour l'analyse détaillée des KPIs
        
        Args:
            analysis_json: Dictionnaire contenant les données d'analyse
        
        Returns:
            str: Prompt formaté pour le LLM
        """
        # Extraire les KPIs
        kpis = analysis_json.get('kpis', {})
        temp_stats = analysis_json.get('statistiques_temperature', {})
        vent_stats = analysis_json.get('statistiques_vitesse_vent', {})
        hum_stats = analysis_json.get('statistiques_humidite', {})
        power_stats = analysis_json.get('statistiques_power', {})
        wind_rose = analysis_json.get('wind_rose_data', {})
        resume = analysis_json.get('resume_general', {})
        
        prompt = f"""
Tu es un expert en analyse de données météorologiques et opérationnelles pour des sites industriels.

CONTEXTE:
Période d'analyse: {resume.get('periode_analyse', 'Non spécifiée')}
Durée: {resume.get('duree_jours', 0)} jours
Nombre de mesures: {resume.get('nb_mesures_recues', 0)}
Qualité des données: {resume.get('qualite_donnees', 'Non spécifiée')}

STATISTIQUES DÉTAILLÉES:

1. TEMPÉRATURE:
{json.dumps(temp_stats, indent=2, default=str, ensure_ascii=False)}

2. VITESSE DU VENT:
{json.dumps(vent_stats, indent=2, default=str, ensure_ascii=False)}

3. HUMIDITÉ:
{json.dumps(hum_stats, indent=2, default=str, ensure_ascii=False)}

4. PUISSANCE:
{json.dumps(power_stats, indent=2, default=str, ensure_ascii=False)}

5. ROSE DES VENTS:
Direction dominante: {wind_rose.get('direction_dominante', {}).get('nom', 'N/A')}
{json.dumps(wind_rose.get('secteurs', [])[:5], indent=2, default=str, ensure_ascii=False) if wind_rose.get('secteurs') else 'Données non disponibles'}

6. INDICATEURS CLÉS (KPIs):
{json.dumps(kpis, indent=2, default=str, ensure_ascii=False)}

TÂCHE:
Génère une analyse PROFONDE et DÉTAILLÉE des KPIs et des données météorologiques. 
Fournis UNIQUEMENT un JSON valide avec cette structure exacte:

{{
    "analyse_detaillee_kpis": {{
        "synthese_generale": "Synthèse générale de 4-6 phrases sur l'état global des KPIs et des conditions opérationnelles",
        "analyse_temps_conditions_optimales": {{
            "valeur": {kpis.get('temps_conditions_optimales', {}).get('pourcentage', 0) if isinstance(kpis.get('temps_conditions_optimales'), dict) else 0},
            "interpretation": "Interprétation détaillée (6-10 phrases) de ce pourcentage avec comparaison aux objectifs, implications opérationnelles, et contexte météorologique",
            "classification": "excellent/bon/moyen/faible",
            "facteurs_influents": ["Facteur 1 expliquant cette valeur", "Facteur 2", "Facteur 3"],
            "impact_operationnel": "Impact détaillé sur les opérations (3-5 phrases)"
        }},
        "analyse_alertes_vent_fort": {{
            "nombre": {kpis.get('nb_alertes_vent_fort', 0)},
            "frequence": "Fréquence calculée (par jour/heure) et interprétation",
            "interpretation": "Analyse détaillée (5-8 phrases) de la fréquence des alertes, sévérité, patterns temporels, et risques associés",
            "niveau_risque": "faible/moyen/élevé/critique",
            "recommandations_vent": ["Recommandation 1 spécifique et actionnable", "Recommandation 2", "Recommandation 3"]
        }},
        "analyse_alertes_humidite": {{
            "nombre": {kpis.get('nb_alertes_humidite', 0)},
            "frequence": "Fréquence calculée et interprétation",
            "interpretation": "Analyse détaillée (5-8 phrases) des épisodes d'humidité élevée, corrélations avec autres paramètres, et impacts",
            "niveau_risque": "faible/moyen/élevé/critique",
            "recommandations_humidite": ["Recommandation 1 spécifique", "Recommandation 2", "Recommandation 3"]
        }},
        "analyse_correlations": {{
            "correlation_temp_vent": "Analyse de la corrélation température-vent (3-4 phrases)",
            "correlation_humidite_puissance": "Analyse de la corrélation humidité-puissance (3-4 phrases)",
            "patterns_identifies": ["Pattern 1 identifié", "Pattern 2", "Pattern 3"],
            "insights": "Insights dérivés des corrélations (4-6 phrases)"
        }},
        "analyse_tendances": {{
            "tendances_principales": ["Tendance 1 avec détails", "Tendance 2", "Tendance 3"],
            "evolution_attendre": "Évolution attendue dans les prochaines périodes (4-5 phrases)",
            "points_attention": ["Point d'attention 1", "Point d'attention 2", "Point d'attention 3"]
        }},
        "anomalies_detectees": [
            "Anomalie 1 avec description détaillée et valeurs",
            "Anomalie 2 si applicable",
            "Anomalie 3 si applicable"
        ],
        "forces_identifiees": [
            "Force 1 avec justification quantitative",
            "Force 2",
            "Force 3"
        ],
        "faiblesses_identifiees": [
            "Faiblesse 1 avec impact quantifié",
            "Faiblesse 2",
            "Faiblesse 3"
        ]
    }},
    "conclusion_generale": {{
        "resume_executif": "Résumé exécutif de 5-7 phrases pour la direction",
        "evaluation_globale": "Évaluation globale de la performance (excellent/bon/moyen/à améliorer) avec justification",
        "priorites_action": [
            {{
                "priorite": 1,
                "action": "Action prioritaire 1",
                "justification": "Justification détaillée avec valeurs et impacts",
                "delai_recommande": "Délai recommandé"
            }},
            {{
                "priorite": 2,
                "action": "Action prioritaire 2",
                "justification": "Justification détaillée",
                "delai_recommande": "Délai recommandé"
            }},
            {{
                "priorite": 3,
                "action": "Action prioritaire 3",
                "justification": "Justification détaillée",
                "delai_recommande": "Délai recommandé"
            }}
        ],
        "recommandations_strategiques": [
            "Recommandation stratégique 1 avec justification",
            "Recommandation stratégique 2",
            "Recommandation stratégique 3"
        ],
        "perspectives_futures": "Perspectives et attentes pour les prochaines périodes (4-6 phrases)",
        "message_cle": "Message clé en une phrase pour la direction"
    }}
}}

IMPORTANT:
- Utilise UNIQUEMENT les valeurs numériques exactes fournies dans les données
- Sois précis et quantitatif dans tes analyses
- Fournis des recommandations actionnables et spécifiques
- Retourne UNIQUEMENT du JSON valide, sans texte avant ou après
- Utilise des phrases complètes et détaillées dans chaque section
- Assure-toi que toutes les valeurs numériques correspondent exactement aux données fournies
"""
        return prompt
    
    def analyze_kpis_with_llm(self, analysis_json):
        """
        Analyse les KPIs en utilisant Gemini LLM
        
        Args:
            analysis_json: Dictionnaire contenant les données d'analyse
        
        Returns:
            dict: Analyse détaillée des KPIs avec conclusion
        """
        if not self.model:
            raise Exception("Modèle Gemini non initialisé")
        
        prompt = self.prepare_kpi_analysis_prompt(analysis_json)
        
        try:
            # Appel à Gemini
            response = self.model.generate_content(
                prompt,
                generation_config={
                    "temperature": 0.3,
                    "top_p": 0.95,
                    "top_k": 40,
                    "max_output_tokens": 4000,
                }
            )
            
            # Extraire le texte de la réponse
            response_text = response.text.strip()
            
            # Parser le JSON de la réponse
            analysis_result = self._extract_json_from_response(response_text)
            
            return analysis_result
            
        except Exception as e:
            st.error(f"Erreur lors de l'analyse LLM: {str(e)}")
            return None
    
    def _extract_json_from_response(self, text):
        """
        Extrait le JSON de la réponse du LLM
        
        Args:
            text: Texte de la réponse
        
        Returns:
            dict: JSON parsé
        """
        import re
        
        # Méthode 1: Chercher entre balises markdown
        json_match = re.search(r'```json\s*(\{.*?\})\s*```', text, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except:
                pass
        
        # Méthode 2: Chercher entre triple backticks simples
        json_match = re.search(r'```\s*(\{.*?\})\s*```', text, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except:
                pass
        
        # Méthode 3: Chercher le premier JSON valide
        start_idx = text.find('{')
        if start_idx != -1:
            # Compter les accolades pour trouver la fin
            brace_count = 0
            json_start = start_idx
            json_end = start_idx
            
            for i in range(start_idx, len(text)):
                if text[i] == '{':
                    brace_count += 1
                elif text[i] == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        json_end = i + 1
                        break
            
            if brace_count == 0 and json_end > json_start:
                json_str = text[json_start:json_end]
                try:
                    return json.loads(json_str)
                except:
                    pass
        
        # Méthode 4: Essayer de parser tout le texte
        try:
            return json.loads(text)
        except:
            pass
        
        raise ValueError("Impossible d'extraire un JSON valide de la réponse")
    
    def merge_analyses(self, original_analysis_json, llm_analysis_result):
        """
        Fusionne l'analyse originale avec l'analyse LLM complémentaire
        
        Args:
            original_analysis_json: Analyse originale du code principal
            llm_analysis_result: Résultat de l'analyse LLM complémentaire
        
        Returns:
            dict: Analyse fusionnée
        """
        merged = original_analysis_json.copy()
        
        # Ajouter l'analyse LLM complémentaire
        merged['analyse_llm_complementaire'] = {
            'timestamp': datetime.now().isoformat(),
            'source': 'Gemini Pro',
            'analyse_detaillee_kpis': llm_analysis_result.get('analyse_detaillee_kpis', {}),
            'conclusion_generale': llm_analysis_result.get('conclusion_generale', {})
        }
        
        return merged
    
    def save_merged_analysis(self, merged_analysis, output_path):
        """
        Sauvegarde l'analyse fusionnée dans un fichier JSON
        
        Args:
            merged_analysis: Analyse fusionnée
            output_path: Chemin de sortie
        """
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(merged_analysis, f, indent=2, ensure_ascii=False, default=str)
    
    def generate_markdown_report(self, llm_analysis_result):
        """
        Génère un rapport Markdown à partir de l'analyse LLM
        
        Args:
            llm_analysis_result: Résultat de l'analyse LLM
        
        Returns:
            str: Rapport Markdown formaté
        """
        lines = []
        lines.append("# Analyse Détaillée des KPIs - Perspective LLM Complémentaire")
        lines.append("")
        lines.append(f"*Généré le {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}*")
        lines.append("")
        
        # Analyse détaillée des KPIs
        kpi_analysis = llm_analysis_result.get('analyse_detaillee_kpis', {})
        
        lines.append("## 1. Synthèse Générale")
        lines.append("")
        lines.append(kpi_analysis.get('synthese_generale', 'Non disponible'))
        lines.append("")
        
        # Conditions optimales
        lines.append("## 2. Analyse du Temps en Conditions Optimales")
        lines.append("")
        opt_analysis = kpi_analysis.get('analyse_temps_conditions_optimales', {})
        lines.append(f"**Valeur:** {opt_analysis.get('valeur', 0):.2f}%")
        lines.append(f"**Classification:** {opt_analysis.get('classification', 'N/A')}")
        lines.append("")
        lines.append("**Interprétation:**")
        lines.append(opt_analysis.get('interpretation', 'Non disponible'))
        lines.append("")
        lines.append("**Facteurs influents:**")
        for facteur in opt_analysis.get('facteurs_influents', []):
            lines.append(f"- {facteur}")
        lines.append("")
        lines.append("**Impact opérationnel:**")
        lines.append(opt_analysis.get('impact_operationnel', 'Non disponible'))
        lines.append("")
        
        # Alertes vent fort
        lines.append("## 3. Analyse des Alertes Vent Fort")
        lines.append("")
        vent_analysis = kpi_analysis.get('analyse_alertes_vent_fort', {})
        lines.append(f"**Nombre d'alertes:** {vent_analysis.get('nombre', 0)}")
        lines.append(f"**Fréquence:** {vent_analysis.get('frequence', 'N/A')}")
        lines.append(f"**Niveau de risque:** {vent_analysis.get('niveau_risque', 'N/A')}")
        lines.append("")
        lines.append("**Interprétation:**")
        lines.append(vent_analysis.get('interpretation', 'Non disponible'))
        lines.append("")
        lines.append("**Recommandations:**")
        for rec in vent_analysis.get('recommandations_vent', []):
            lines.append(f"- {rec}")
        lines.append("")
        
        # Alertes humidité
        lines.append("## 4. Analyse des Alertes Humidité")
        lines.append("")
        hum_analysis = kpi_analysis.get('analyse_alertes_humidite', {})
        lines.append(f"**Nombre d'alertes:** {hum_analysis.get('nombre', 0)}")
        lines.append(f"**Fréquence:** {hum_analysis.get('frequence', 'N/A')}")
        lines.append(f"**Niveau de risque:** {hum_analysis.get('niveau_risque', 'N/A')}")
        lines.append("")
        lines.append("**Interprétation:**")
        lines.append(hum_analysis.get('interpretation', 'Non disponible'))
        lines.append("")
        lines.append("**Recommandations:**")
        for rec in hum_analysis.get('recommandations_humidite', []):
            lines.append(f"- {rec}")
        lines.append("")
        
        # Corrélations
        lines.append("## 5. Analyse des Corrélations")
        lines.append("")
        corr_analysis = kpi_analysis.get('analyse_correlations', {})
        lines.append("**Corrélation Température-Vent:**")
        lines.append(corr_analysis.get('correlation_temp_vent', 'Non disponible'))
        lines.append("")
        lines.append("**Corrélation Humidité-Puissance:**")
        lines.append(corr_analysis.get('correlation_humidite_puissance', 'Non disponible'))
        lines.append("")
        lines.append("**Patterns identifiés:**")
        for pattern in corr_analysis.get('patterns_identifies', []):
            lines.append(f"- {pattern}")
        lines.append("")
        lines.append("**Insights:**")
        lines.append(corr_analysis.get('insights', 'Non disponible'))
        lines.append("")
        
        # Tendances
        lines.append("## 6. Analyse des Tendances")
        lines.append("")
        trend_analysis = kpi_analysis.get('analyse_tendances', {})
        lines.append("**Tendances principales:**")
        for trend in trend_analysis.get('tendances_principales', []):
            lines.append(f"- {trend}")
        lines.append("")
        lines.append("**Évolution attendue:**")
        lines.append(trend_analysis.get('evolution_attendre', 'Non disponible'))
        lines.append("")
        lines.append("**Points d'attention:**")
        for point in trend_analysis.get('points_attention', []):
            lines.append(f"- {point}")
        lines.append("")
        
        # Anomalies, forces, faiblesses
        lines.append("## 7. Anomalies, Forces et Faiblesses")
        lines.append("")
        lines.append("### Anomalies détectées:")
        for anom in kpi_analysis.get('anomalies_detectees', []):
            lines.append(f"- {anom}")
        lines.append("")
        lines.append("### Forces identifiées:")
        for force in kpi_analysis.get('forces_identifiees', []):
            lines.append(f"- {force}")
        lines.append("")
        lines.append("### Faiblesses identifiées:")
        for faiblesse in kpi_analysis.get('faiblesses_identifiees', []):
            lines.append(f"- {faiblesse}")
        lines.append("")
        
        # Conclusion générale
        lines.append("## 8. Conclusion Générale")
        lines.append("")
        conclusion = llm_analysis_result.get('conclusion_generale', {})
        lines.append("### Résumé Exécutif")
        lines.append("")
        lines.append(conclusion.get('resume_executif', 'Non disponible'))
        lines.append("")
        lines.append(f"### Évaluation Globale: {conclusion.get('evaluation_globale', 'N/A')}")
        lines.append("")
        lines.append("### Priorités d'Action")
        lines.append("")
        for priorite in conclusion.get('priorites_action', []):
            lines.append(f"#### Priorité {priorite.get('priorite', 'N/A')}: {priorite.get('action', 'N/A')}")
            lines.append("")
            lines.append(f"**Justification:** {priorite.get('justification', 'N/A')}")
            lines.append("")
            lines.append(f"**Délai recommandé:** {priorite.get('delai_recommande', 'N/A')}")
            lines.append("")
        lines.append("### Recommandations Stratégiques")
        lines.append("")
        for rec in conclusion.get('recommandations_strategiques', []):
            lines.append(f"- {rec}")
        lines.append("")
        lines.append("### Perspectives Futures")
        lines.append("")
        lines.append(conclusion.get('perspectives_futures', 'Non disponible'))
        lines.append("")
        lines.append("### Message Clé")
        lines.append("")
        lines.append(f"**{conclusion.get('message_cle', 'N/A')}**")
        lines.append("")
        lines.append("---")
        lines.append("*Analyse générée par Gemini Pro - Perspective complémentaire*")
        
        return "\n".join(lines)


def main():
    """
    Fonction principale pour exécuter l'analyse complémentaire
    Peut être utilisée en ligne de commande ou intégrée dans Streamlit
    """
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python analyse_kpi_llm.py <chemin_vers_json> [chemin_sortie]")
        print("Ou utilisez la fonction dans Streamlit avec analyse_json directement")
        return
    
    json_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else "analyse_kpi_fusionnee.json"
    
    analyzer = KPIAnalyzerLLM()
    
    # Charger les données
    analysis_json = analyzer.load_json_data(json_path=json_path)
    
    # Analyser avec LLM
    print("Analyse des KPIs en cours avec Gemini...")
    llm_analysis = analyzer.analyze_kpis_with_llm(analysis_json)
    
    if llm_analysis:
        # Fusionner les analyses
        merged = analyzer.merge_analyses(analysis_json, llm_analysis)
        
        # Sauvegarder
        analyzer.save_merged_analysis(merged, output_path)
        print(f"Analyse fusionnée sauvegardée dans: {output_path}")
        
        # Générer le rapport Markdown
        markdown_report = analyzer.generate_markdown_report(llm_analysis)
        report_path = output_path.replace('.json', '_rapport.md')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(markdown_report)
        print(f"Rapport Markdown sauvegardé dans: {report_path}")
    else:
        print("Erreur lors de l'analyse LLM")


if __name__ == "__main__":
    main()


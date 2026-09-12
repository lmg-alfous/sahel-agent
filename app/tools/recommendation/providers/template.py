"""Deterministic recommendation synthesis from structured tool outputs.

IMPLEMENTED. No LLM. Maps risk sub-scores + anomalies to prioritised actions
using transparent rules. Always attaches limitations and a non-diagnosis warning.
"""
from __future__ import annotations

from typing import Any, Dict, List

from pydantic import BaseModel

from app.core.schemas import HealthState, HealthStatus, MaturityStatus
from app.tools.base import BaseProvider
from app.tools.recommendation.schema import RecommendationInput, RecommendationOutput


def _risk_level(risk: Dict[str, Any] | None, key: str) -> str:
    if not risk:
        return "unknown"
    sub = risk.get(key) or {}
    return sub.get("level", "unknown")


def synthesize(payload: RecommendationInput) -> Dict[str, Any]:
    risk = payload.risk or {}
    combined = _risk_level(risk, "combined_risk")
    water = _risk_level(risk, "water_stress")
    heat = _risk_level(risk, "heat_stress")
    env = _risk_level(risk, "environmental")

    actions: List[str] = []
    monitoring: List[str] = []
    warnings: List[str] = []

    french = payload.language in {"fr", "bm"}
    if water == "high":
        actions.append(
            "Priorisez une irrigation ciblée dans les 24 h ; vérifiez l'humidité du sol "
            "avec une sonde à profondeur de racines avant et après."
            if french else "Prioritise targeted irrigation within the next 24h; verify soil moisture with a probe at root depth before and after."
        )
    elif water == "moderate":
        actions.append("Prévoyez une irrigation complémentaire dans les 1 à 2 prochains jours si aucune pluie n'est prévue." if french else "Plan supplemental irrigation in the next 1-2 days if no rain is forecast.")

    if heat == "high":
        actions.append("Réduisez la charge de chaleur à midi si possible (horaire d'irrigation, ombrage ou paillage) ; évitez les opérations qui ajoutent du stress pendant les pics de chaleur." if french else "Reduce midday heat load where feasible (irrigation timing, shading/mulch); avoid operations that add plant stress during peak heat.")
    elif heat == "moderate":
        actions.append("Planifiez le travail au champ tôt le matin ou le soir pour limiter l'exposition à la chaleur." if french else "Schedule field work for early morning/evening to limit heat exposure.")

    if env == "high":
        actions.append("Inspectez la pression des maladies et les problèmes de drainage ; ajustez la gestion du couvert végétal si nécessaire." if french else "Inspect for disease pressure and drainage issues; adjust canopy management as needed.")
    elif env == "moderate":
        actions.append("Augmentez la fréquence des observations pour repérer tôt les symptômes de maladie ou de carence." if french else "Increase scouting frequency for early disease or nutrient symptoms.")

    # Vision-driven note.
    vsigns = (payload.vision or {}).get("possible_signs") or []
    if vsigns:
        monitoring.append(
            "Re-photograph the same plants in 48-72h under similar light to track the "
            f"possible sign(s): {', '.join(vsigns[:3])}."
        )

    # Sensor anomalies -> monitoring.
    for anomaly in (payload.sensors or {}).get("anomalies", [])[:4]:
        monitoring.append(f"Track: {anomaly}.")

    # Weather-driven note.
    wx = payload.weather or {}
    if isinstance(wx.get("rain_probability"), (int, float)):
        monitoring.append(
            f"Weather ({wx.get('source', 'n/a')}): rain probability ~{wx['rain_probability']}%; "
            "revisit irrigation plan if this changes."
        )

    # Cross-check surfacing.
    xc = risk.get("cross_check", {})
    for conv in xc.get("converging_evidence", []):
        warnings.append(f"Converging evidence: {conv}.")
    for div in xc.get("diverging_evidence", []):
        warnings.append(f"Note (conflicting signals): {div}")

    if not actions:
        actions.append("Les heuristiques du prototype n'indiquent pas d'intervention urgente ; maintenez le plan habituel." if french else "No urgent intervention indicated by the prototype heuristics; maintain the normal plan.")
    if not monitoring:
        monitoring.append("Continue routine monitoring of soil moisture, temperature and canopy condition.")

    priority = {"high": "high", "moderate": "medium", "low": "low", "unknown": "low"}[combined]

    xc_div = (risk.get("cross_check", {}) or {}).get("diverging_evidence", [])
    if combined == "unknown":
        priority = "low"
    elif xc_div and priority == "low":
        # Contradictory signals -> don't let a 'low' combined score read as "all clear".
        priority = "medium"

    confidence = float(risk.get("confidence", 0.4))
    main_finding = _main_finding(combined, water, heat, env, payload)
    if combined == "unknown":
        main_finding = ("Éléments insuffisants pour estimer le risque environnemental : aucune donnée de capteur, image ou donnée météo utilisable. Fournissez au moins des relevés de capteurs ou une image." if french else "Insufficient evidence to estimate environmental risk: no sensor, image or weather data was usable. Provide at least sensor readings or an image.")
    elif xc_div:
        main_finding = (
            main_finding
            + " Sources do not fully converge (" + xc_div[0].rstrip(".")
            + ") — treat this result as provisional and confirm on site."
        )

    warnings.append("Ceci est une aide à la décision, pas un conseil agronomique. Les chiffres sont des indicateurs de prototype et doivent être confirmés sur place par une personne qualifiée." if french else "This is decision SUPPORT, not agronomic advice. Figures are prototype indicators and require confirmation by a qualified person on site.")

    limitations = [
        "Heuristic rules, not a validated model; thresholds need local calibration.",
        f"Based on modalities: {', '.join(payload.modalities) or 'none'}.",
    ]
    if (payload.sensors or {}).get("missing_fields"):
        limitations.append(
            "Missing sensor inputs: " + ", ".join(payload.sensors["missing_fields"]) + "."
        )

    evidence = (payload.evidence or {}).get("results") or []
    evidence_source = (payload.evidence or {}).get("source", "")
    if evidence:
        if evidence_source == "exa":
            actions.append(
                "Cross-check the situation against the retrieved advisory sources "
                "(Evidence section) before acting."
            )
        else:
            limitations.append(
                "Evidence section shows offline sample sources, not live search results "
                "(set EXA_API_KEY for real citations)."
            )

    return RecommendationOutput(
        priority=priority,
        main_finding=main_finding,
        recommended_actions=actions,
        monitoring_actions=monitoring,
        warnings=warnings,
        confidence=round(confidence, 2),
        limitations=limitations,
        evidence=evidence,
        evidence_source=evidence_source,
        method="deterministic template",
    ).model_dump()


def _main_finding(combined: str, water: str, heat: str, env: str, payload: RecommendationInput) -> str:
    french = payload.language in {"fr", "bm"}
    if combined == "unknown":
        return "Insufficient data to estimate combined environmental risk with confidence."
    drivers = []
    if water in {"moderate", "high"}:
        drivers.append(f"stress hydrique ({_fr_level(water)})" if french else f"water stress ({water})")
    if heat in {"moderate", "high"}:
        drivers.append(f"stress thermique ({_fr_level(heat)})" if french else f"heat stress ({heat})")
    if env in {"moderate", "high"}:
        drivers.append(f"risque environnemental ({_fr_level(env)})" if french else f"environmental risk ({env})")
    driver_txt = ", ".join(drivers) if drivers else ("aucun facteur dominant unique" if french else "no single dominant driver")
    stage = payload.growth_stage
    if french:
        stage_txt = f" au stade {stage}" if stage and stage != "unknown" else ""
        return f"Le risque environnemental combiné est {_fr_level(combined)}{stage_txt} ; principaux facteurs : {driver_txt}."
    stage_txt = f" at the {stage} stage" if stage and stage != "unknown" else ""
    return f"Combined environmental risk is {combined}{stage_txt}; main contributor(s): {driver_txt}."


def _fr_level(level: str) -> str:
    return {"low": "faible", "moderate": "modéré", "medium": "moyen", "high": "élevé", "unknown": "inconnu"}.get(level, level)


class TemplateRecommendationProvider(BaseProvider):
    name = "template"
    status = MaturityStatus.IMPLEMENTED

    def run(self, payload: BaseModel) -> Dict[str, Any]:
        assert isinstance(payload, RecommendationInput)
        return synthesize(payload)

    def health_check(self) -> HealthStatus:
        return HealthStatus(
            component="provider:template",
            state=HealthState.READY,
            detail="deterministic recommendation synthesis",
            provider=self.name,
        )

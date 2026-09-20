"""Farm-scale model: many string inverters, array size in the client's hands.

Two inverter classes, both from E:\\swarm\\inverter-classes.json, nameless by
rule. Their provenance travels with them: the 352 kVA class is FROM-DATASHEET
and in volume production; the larger class's nearest real product is a 465 kW
machine announced January 2026 and is FROM-PRESS only - no datasheet has been
published, and its DC input figures in that file are marked ASSUMED. It is
modelled here as what it is, not as a 500 kVA datasheet.

    python farm.py                      the reference farm, 1000 inverters
    python farm.py --inverters 900 --series 28 --strings 22
    python farm.py --class large --solve-dcac 1.35

Every figure printed is computed from the arguments. No yield, no energy, no
export figure is stated: none is computed here.
"""
import argparse
import json
import sys

# ---- the two classes, as owned. AC kVA, max strings, provenance, note
CLASSES = {
    "352": {
        "label": "352 kVA class, 1500 V string inverter, variant A",
        "kva": 352.0,
        "max_strings": 24,             # 12 MPPT x 2 inputs
        "mppt": 12, "inputs_per_mppt": 2,
        "src": "FROM-DATASHEET",
        "note": "real, in volume production, fully documented",
    },
    "500": {
        "label": "500 kW class string inverter",
        "kva": 500.0,
        # scaled, not invented: the string is fixed by the 1500 V ceiling, so a
        # larger machine changes only how many strings hang off it.
        "max_strings": 34,             # = floor(1.35 x 500 / 19.8), the ratio
        "mppt": 0, "inputs_per_mppt": 0,
        "src": "SEEN-BY-OWNER",
        "note": "a 500 kW machine was seen on a stand at a European exhibition "
                "in 2026. Existence is first hand; the specification is not. "
                "MPPT count and inputs per MPPT were not captured, so the "
                "string count here is the ratio, not the machine's input list.",
    },
    "large": {
        "label": "465 kW class string inverter (the 500 kVA slot)",
        "kva": 465.0,
        "max_strings": 36,             # 6 MPPT x 6 inputs
        "mppt": 6, "inputs_per_mppt": 6,
        "src": "FROM-PRESS",
        "note": "announced January 2026, press only - NO DATASHEET PUBLISHED. "
                "Its DC input currents are ASSUMED. Do not commit against it.",
    },
}
MODULE_WP = 660.0            # class T660, MODULE-CLASSES.md
DCAC_LO, DCAC_HI = 1.15, 1.40


def farm(cls, inverters, series, strings, wp=MODULE_WP):
    c = CLASSES[cls]
    if strings > c["max_strings"]:
        raise SystemExit(
            "REFUSED: %d strings exceeds this class's %d (%d MPPT x %d "
            "inputs). The inverter cannot accept them."
            % (strings, c["max_strings"], c["mppt"], c["inputs_per_mppt"]))
    kw_string = series * wp / 1000.0
    kw_inv = kw_string * strings
    return {
        "class": c["label"], "provenance": c["src"], "note": c["note"],
        "inverters": inverters,
        "modules_in_series": series, "strings_per_inverter": strings,
        "module_wp": wp,
        "kw_dc_per_string": round(kw_string, 3),
        "kw_dc_per_inverter": round(kw_inv, 2),
        "dc_ac_per_inverter": round(kw_inv / c["kva"], 4),
        "mated_pairs_per_inverter": strings * 2,
        "mw_dc_farm": round(kw_inv * inverters / 1000.0, 2),
        "mva_ac_farm": round(c["kva"] * inverters / 1000.0, 2),
        "strings_farm": strings * inverters,
        "modules_farm": strings * series * inverters,
        "mated_pairs_farm": strings * 2 * inverters,
        "in_band": DCAC_LO <= kw_inv / c["kva"] <= DCAC_HI,
    }


def solve_strings(cls, series, target, wp=MODULE_WP):
    """Most strings that stay at or under a target DC:AC, within the class."""
    c = CLASSES[cls]
    kw_string = series * wp / 1000.0
    n = int(target * c["kva"] / kw_string)
    return min(n, c["max_strings"]), n > c["max_strings"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--class", dest="cls", default="352",
                    choices=sorted(CLASSES))
    ap.add_argument("--inverters", type=int, default=1000)
    ap.add_argument("--series", type=int, default=30)
    ap.add_argument("--strings", type=int, default=24)
    ap.add_argument("--wp", type=float, default=MODULE_WP)
    ap.add_argument("--solve-dcac", type=float, default=None)
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()

    strings = a.strings
    capped = False
    if a.solve_dcac:
        strings, capped = solve_strings(a.cls, a.series, a.solve_dcac, a.wp)

    f = farm(a.cls, a.inverters, a.series, strings, a.wp)
    if a.json:
        print(json.dumps(f, indent=1)); return 0

    print("FARM MODEL  %s" % f["class"])
    print("  provenance   %s - %s" % (f["provenance"], f["note"]))
    if f["provenance"] != "FROM-DATASHEET":
        print("  *** this class is not documented. Treat every number below "
              "as indicative. ***")
    print()
    print("  ARRAY, in your hands")
    print("    modules in series per string ... %d" % f["modules_in_series"])
    print("    strings per inverter .......... %d of a maximum %d"
          % (f["strings_per_inverter"], CLASSES[a.cls]["max_strings"]))
    print("    module ........................ %.0f Wp" % f["module_wp"])
    if a.solve_dcac:
        print("    (strings solved for a DC:AC target of %.2f%s)"
              % (a.solve_dcac,
                 "; CAPPED by the inverter's input count" if capped else ""))
    print()
    print("  ONE INVERTER")
    print("    kW DC per string .............. %8.2f kW" % f["kw_dc_per_string"])
    print("    kW DC per inverter ............ %8.2f kW" % f["kw_dc_per_inverter"])
    print("    DC:AC ......................... %8.3f   %s"
          % (f["dc_ac_per_inverter"],
             "in the %.2f-%.2f band" % (DCAC_LO, DCAC_HI) if f["in_band"]
             else "OUTSIDE the %.2f-%.2f band" % (DCAC_LO, DCAC_HI)))
    print("    mated pairs, inverter side .... %8d" % f["mated_pairs_per_inverter"])
    print()
    print("  THE FARM, %d inverters" % f["inverters"])
    print("    DC .............. %10.2f MWp" % f["mw_dc_farm"])
    print("    AC .............. %10.2f MVA" % f["mva_ac_farm"])
    print("    strings ......... %10s" % "{:,}".format(f["strings_farm"]))
    print("    modules ......... %10s" % "{:,}".format(f["modules_farm"]))
    print("    mated pairs ..... %10s   (inverter side only)"
          % "{:,}".format(f["mated_pairs_farm"]))
    print()
    print("  Not computed, and not implied: yield, energy, export, land area,")
    print("  cable schedule or cost. None of those is in this model.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

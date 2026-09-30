#!/usr/bin/env python3
"""
Fictional knowledge base and tool functions for the kb_tool benchmark.

Provides:
  - KB: ~50 fictional entities with relationships (people, companies, products, awards, universities, cities)
  - search_kb(query): fuzzy match across entity names/descriptions, return top 3 summaries
  - lookup_kb(entity_id): return full JSON details for one entity
  - calculate(expression): safe math eval
  - patch_agent_tools(mod): monkey-patch a generated agent module's tool_* functions
"""

import json
import math
import re

# ── Knowledge Base ──────────────────────────────────────────────

CITIES = {
    "city_portland": {
        "id": "city_portland", "type": "city",
        "name": "Portland", "country": "United States", "population": 652503,
        "description": "A major city in Oregon known for its tech industry and green spaces.",
    },
    "city_zurich": {
        "id": "city_zurich", "type": "city",
        "name": "Zurich", "country": "Switzerland", "population": 434008,
        "description": "The largest city in Switzerland, a global financial center.",
    },
    "city_nairobi": {
        "id": "city_nairobi", "type": "city",
        "name": "Nairobi", "country": "Kenya", "population": 4397073,
        "description": "The capital of Kenya and a major hub for East African technology.",
    },
    "city_osaka": {
        "id": "city_osaka", "type": "city",
        "name": "Osaka", "country": "Japan", "population": 2753862,
        "description": "Japan's second largest city, known for manufacturing and food culture.",
    },
    "city_halifax": {
        "id": "city_halifax", "type": "city",
        "name": "Halifax", "country": "Canada", "population": 439819,
        "description": "The capital of Nova Scotia, a port city on Canada's Atlantic coast.",
    },
}

UNIVERSITIES = {
    "uni_westfield": {
        "id": "uni_westfield", "type": "university",
        "name": "Westfield Institute of Technology", "location": "city_portland",
        "founded_year": 1962,
        "description": "A private research university in Portland specializing in engineering and AI.",
    },
    "uni_lakeview": {
        "id": "uni_lakeview", "type": "university",
        "name": "Lakeview University", "location": "city_zurich",
        "founded_year": 1891,
        "description": "A prestigious Swiss university known for its physics and mathematics programs.",
    },
    "uni_savannah": {
        "id": "uni_savannah", "type": "university",
        "name": "Savannah College of Science", "location": "city_nairobi",
        "founded_year": 2003,
        "description": "A young but fast-growing university in Nairobi focused on applied sciences.",
    },
    "uni_cedar": {
        "id": "uni_cedar", "type": "university",
        "name": "Cedar Valley University", "location": "city_halifax",
        "founded_year": 1934,
        "description": "A Canadian liberal arts university in Halifax with strong marine biology programs.",
    },
    "uni_kanto": {
        "id": "uni_kanto", "type": "university",
        "name": "Kanto Technical University", "location": "city_osaka",
        "founded_year": 1948,
        "description": "A Japanese engineering university in Osaka known for robotics research.",
    },
}

AWARDS = {
    "award_zenith": {
        "id": "award_zenith", "type": "award",
        "name": "Zenith Prize for Innovation", "organization": "Global Tech Foundation",
        "established_year": 2005,
        "description": "An annual award for breakthrough technology products, with a $500,000 prize.",
    },
    "award_meridian": {
        "id": "award_meridian", "type": "award",
        "name": "Meridian Research Medal", "organization": "International Science Council",
        "established_year": 1978,
        "description": "A biennial medal recognizing outstanding contributions to applied research.",
    },
    "award_nova": {
        "id": "award_nova", "type": "award",
        "name": "Nova Engineering Award", "organization": "Pacific Engineering Society",
        "established_year": 2012,
        "description": "Recognizes engineers under 40 for exceptional design achievements.",
    },
    "award_atlas": {
        "id": "award_atlas", "type": "award",
        "name": "Atlas Humanitarian Tech Prize", "organization": "Atlas Foundation",
        "established_year": 2015,
        "description": "Awarded to technology projects that improve quality of life in developing regions.",
    },
    "award_pinnacle": {
        "id": "award_pinnacle", "type": "award",
        "name": "Pinnacle Business Award", "organization": "World Commerce Forum",
        "established_year": 1999,
        "description": "Annual award for the most impactful business innovation globally.",
    },
}

PEOPLE = {
    "person_elena": {
        "id": "person_elena", "type": "person",
        "name": "Dr. Elena Voss", "born": 1985,
        "education": {"university": "uni_lakeview", "degree": "PhD Physics", "year": 2012},
        "affiliation": "comp_helios",
        "awards": ["award_meridian"],
        "field": "quantum computing",
        "description": "A quantum computing researcher who founded Helios Quantum.",
    },
    "person_marcus": {
        "id": "person_marcus", "type": "person",
        "name": "Marcus Tanaka", "born": 1979,
        "education": {"university": "uni_kanto", "degree": "MSc Robotics", "year": 2004},
        "affiliation": "comp_aether",
        "awards": ["award_nova", "award_zenith"],
        "field": "robotics",
        "description": "A robotics engineer and founder of Aether Robotics.",
    },
    "person_amara": {
        "id": "person_amara", "type": "person",
        "name": "Amara Okafor", "born": 1991,
        "education": {"university": "uni_savannah", "degree": "BSc Computer Science", "year": 2013},
        "affiliation": "comp_solaris",
        "awards": ["award_atlas"],
        "field": "mobile technology",
        "description": "A software entrepreneur from Kenya who founded Solaris Mobile.",
    },
    "person_james": {
        "id": "person_james", "type": "person",
        "name": "James Whitfield", "born": 1972,
        "education": {"university": "uni_westfield", "degree": "PhD Computer Science", "year": 2000},
        "affiliation": "comp_nexus",
        "awards": ["award_pinnacle"],
        "field": "artificial intelligence",
        "description": "An AI pioneer and CEO of Nexus AI Labs.",
    },
    "person_sophie": {
        "id": "person_sophie", "type": "person",
        "name": "Dr. Sophie Clarkson", "born": 1988,
        "education": {"university": "uni_cedar", "degree": "PhD Marine Biology", "year": 2015},
        "affiliation": "comp_oceanic",
        "awards": [],
        "field": "ocean technology",
        "description": "A marine biologist who co-founded Oceanic Data Systems.",
    },
    "person_raj": {
        "id": "person_raj", "type": "person",
        "name": "Raj Patel", "born": 1983,
        "education": {"university": "uni_westfield", "degree": "MSc Electrical Engineering", "year": 2007},
        "affiliation": "comp_helios",
        "awards": [],
        "field": "quantum hardware",
        "description": "Chief engineer at Helios Quantum, specializing in quantum chip fabrication.",
    },
    "person_lin": {
        "id": "person_lin", "type": "person",
        "name": "Lin Zhao", "born": 1990,
        "education": {"university": "uni_kanto", "degree": "PhD Mechanical Engineering", "year": 2018},
        "affiliation": "comp_aether",
        "awards": ["award_nova"],
        "field": "industrial automation",
        "description": "Lead engineer at Aether Robotics working on warehouse automation.",
    },
    "person_freya": {
        "id": "person_freya", "type": "person",
        "name": "Freya Lindqvist", "born": 1986,
        "education": {"university": "uni_lakeview", "degree": "MSc Mathematics", "year": 2010},
        "affiliation": "comp_nexus",
        "awards": [],
        "field": "machine learning",
        "description": "Head of research at Nexus AI Labs, focused on reinforcement learning.",
    },
    "person_kwame": {
        "id": "person_kwame", "type": "person",
        "name": "Kwame Asante", "born": 1994,
        "education": {"university": "uni_savannah", "degree": "MSc Data Science", "year": 2018},
        "affiliation": "comp_solaris",
        "awards": [],
        "field": "data analytics",
        "description": "CTO of Solaris Mobile, specializing in mobile data infrastructure.",
    },
    "person_olivia": {
        "id": "person_olivia", "type": "person",
        "name": "Olivia Mercer", "born": 1975,
        "education": {"university": "uni_cedar", "degree": "PhD Oceanography", "year": 2003},
        "affiliation": "comp_oceanic",
        "awards": ["award_meridian"],
        "field": "oceanography",
        "description": "Co-founder and CEO of Oceanic Data Systems, expert in deep-sea sensors.",
    },
    "person_diego": {
        "id": "person_diego", "type": "person",
        "name": "Diego Alvarez", "born": 1982,
        "education": {"university": "uni_westfield", "degree": "MBA", "year": 2008},
        "affiliation": "comp_terravolt",
        "awards": ["award_pinnacle"],
        "field": "clean energy",
        "description": "Founder of TerraVolt Energy, a clean energy startup in Portland.",
    },
    "person_nina": {
        "id": "person_nina", "type": "person",
        "name": "Nina Johansson", "born": 1992,
        "education": {"university": "uni_lakeview", "degree": "PhD Chemistry", "year": 2019},
        "affiliation": "comp_terravolt",
        "awards": [],
        "field": "battery chemistry",
        "description": "Lead scientist at TerraVolt Energy working on solid-state batteries.",
    },
    "person_carlos": {
        "id": "person_carlos", "type": "person",
        "name": "Carlos Rivera", "born": 1978,
        "education": {"university": "uni_kanto", "degree": "BSc Electronics", "year": 2001},
        "affiliation": "comp_aether",
        "awards": [],
        "field": "sensor technology",
        "description": "VP of hardware at Aether Robotics, expert in LiDAR and sensor fusion.",
    },
    "person_yuki": {
        "id": "person_yuki", "type": "person",
        "name": "Yuki Nakamura", "born": 1996,
        "education": {"university": "uni_kanto", "degree": "MSc AI", "year": 2021},
        "affiliation": "comp_nexus",
        "awards": [],
        "field": "natural language processing",
        "description": "Research scientist at Nexus AI Labs working on language models.",
    },
    "person_sarah": {
        "id": "person_sarah", "type": "person",
        "name": "Sarah Mbeki", "born": 1987,
        "education": {"university": "uni_savannah", "degree": "BSc Electrical Engineering", "year": 2010},
        "affiliation": "comp_solaris",
        "awards": ["award_atlas"],
        "field": "solar energy",
        "description": "VP of engineering at Solaris Mobile, pioneered solar-powered base stations.",
    },
}

COMPANIES = {
    "comp_helios": {
        "id": "comp_helios", "type": "company",
        "name": "Helios Quantum", "founded_year": 2016,
        "founder_id": "person_elena",
        "hq_city": "city_zurich",
        "products": ["prod_qubit_x", "prod_qsim"],
        "employees": 340, "revenue": 28000000,
        "description": "A quantum computing company based in Zurich, founded by Dr. Elena Voss.",
    },
    "comp_aether": {
        "id": "comp_aether", "type": "company",
        "name": "Aether Robotics", "founded_year": 2010,
        "founder_id": "person_marcus",
        "hq_city": "city_osaka",
        "products": ["prod_sentinel", "prod_atlas_arm"],
        "employees": 580, "revenue": 95000000,
        "description": "A robotics company in Osaka founded by Marcus Tanaka.",
    },
    "comp_solaris": {
        "id": "comp_solaris", "type": "company",
        "name": "Solaris Mobile", "founded_year": 2015,
        "founder_id": "person_amara",
        "hq_city": "city_nairobi",
        "products": ["prod_sunlink", "prod_datawave"],
        "employees": 210, "revenue": 18000000,
        "description": "A mobile technology company in Nairobi founded by Amara Okafor.",
    },
    "comp_nexus": {
        "id": "comp_nexus", "type": "company",
        "name": "Nexus AI Labs", "founded_year": 2008,
        "founder_id": "person_james",
        "hq_city": "city_portland",
        "products": ["prod_cognis", "prod_neurosync"],
        "employees": 720, "revenue": 150000000,
        "description": "An AI research company in Portland, founded by James Whitfield.",
    },
    "comp_oceanic": {
        "id": "comp_oceanic", "type": "company",
        "name": "Oceanic Data Systems", "founded_year": 2017,
        "founder_id": "person_olivia",
        "hq_city": "city_halifax",
        "products": ["prod_deepscanner", "prod_tidewatch"],
        "employees": 95, "revenue": 8500000,
        "description": "An ocean technology company in Halifax co-founded by Olivia Mercer and Sophie Clarkson.",
    },
    "comp_terravolt": {
        "id": "comp_terravolt", "type": "company",
        "name": "TerraVolt Energy", "founded_year": 2019,
        "founder_id": "person_diego",
        "hq_city": "city_portland",
        "products": ["prod_voltcell", "prod_gridbox"],
        "employees": 165, "revenue": 12000000,
        "description": "A clean energy startup in Portland founded by Diego Alvarez.",
    },
    "comp_vertexai": {
        "id": "comp_vertexai", "type": "company",
        "name": "VertexAI Solutions", "founded_year": 2020,
        "founder_id": "person_freya",
        "hq_city": "city_zurich",
        "products": ["prod_optinet"],
        "employees": 48, "revenue": 3200000,
        "description": "A small AI consulting firm in Zurich co-founded by Freya Lindqvist.",
    },
    "comp_canopy": {
        "id": "comp_canopy", "type": "company",
        "name": "Canopy Drones", "founded_year": 2021,
        "founder_id": "person_kwame",
        "hq_city": "city_nairobi",
        "products": ["prod_skyseed"],
        "employees": 32, "revenue": 1500000,
        "description": "A drone startup in Nairobi founded by Kwame Asante for agricultural monitoring.",
    },
    "comp_polaris": {
        "id": "comp_polaris", "type": "company",
        "name": "Polaris Navigation", "founded_year": 2013,
        "founder_id": "person_carlos",
        "hq_city": "city_osaka",
        "products": ["prod_wayfinder"],
        "employees": 120, "revenue": 22000000,
        "description": "A navigation systems company in Osaka founded by Carlos Rivera.",
    },
    "comp_arctic": {
        "id": "comp_arctic", "type": "company",
        "name": "Arctic Compute", "founded_year": 2022,
        "founder_id": "person_nina",
        "hq_city": "city_halifax",
        "products": ["prod_frostcore"],
        "employees": 25, "revenue": 800000,
        "description": "A data center cooling startup in Halifax founded by Nina Johansson.",
    },
}

PRODUCTS = {
    "prod_qubit_x": {
        "id": "prod_qubit_x", "type": "product",
        "name": "QubitX Processor", "manufacturer_id": "comp_helios",
        "launched_year": 2022, "price": 1200000,
        "awards_won": ["award_zenith"],
        "description": "A 128-qubit quantum processor for research applications.",
    },
    "prod_qsim": {
        "id": "prod_qsim", "type": "product",
        "name": "QSim Cloud Platform", "manufacturer_id": "comp_helios",
        "launched_year": 2023, "price": 50000,
        "awards_won": [],
        "description": "A cloud-based quantum simulation service for enterprises.",
    },
    "prod_sentinel": {
        "id": "prod_sentinel", "type": "product",
        "name": "Sentinel Warehouse Robot", "manufacturer_id": "comp_aether",
        "launched_year": 2018, "price": 85000,
        "awards_won": ["award_zenith", "award_nova"],
        "description": "An autonomous warehouse robot for inventory management and picking.",
    },
    "prod_atlas_arm": {
        "id": "prod_atlas_arm", "type": "product",
        "name": "Atlas Industrial Arm", "manufacturer_id": "comp_aether",
        "launched_year": 2021, "price": 120000,
        "awards_won": [],
        "description": "A high-precision industrial robotic arm for manufacturing.",
    },
    "prod_sunlink": {
        "id": "prod_sunlink", "type": "product",
        "name": "SunLink Base Station", "manufacturer_id": "comp_solaris",
        "launched_year": 2017, "price": 15000,
        "awards_won": ["award_atlas"],
        "description": "A solar-powered mobile base station for rural connectivity.",
    },
    "prod_datawave": {
        "id": "prod_datawave", "type": "product",
        "name": "DataWave Analytics Suite", "manufacturer_id": "comp_solaris",
        "launched_year": 2020, "price": 2400,
        "awards_won": [],
        "description": "A mobile analytics platform for telecom operators.",
    },
    "prod_cognis": {
        "id": "prod_cognis", "type": "product",
        "name": "Cognis Enterprise AI", "manufacturer_id": "comp_nexus",
        "launched_year": 2019, "price": 250000,
        "awards_won": ["award_pinnacle"],
        "description": "An enterprise AI platform for business process automation.",
    },
    "prod_neurosync": {
        "id": "prod_neurosync", "type": "product",
        "name": "NeuroSync Headset", "manufacturer_id": "comp_nexus",
        "launched_year": 2023, "price": 3500,
        "awards_won": ["award_zenith"],
        "description": "A brain-computer interface headset for focus and productivity tracking.",
    },
    "prod_deepscanner": {
        "id": "prod_deepscanner", "type": "product",
        "name": "DeepScanner Sonar Array", "manufacturer_id": "comp_oceanic",
        "launched_year": 2020, "price": 180000,
        "awards_won": ["award_meridian"],
        "description": "An advanced deep-sea sonar system for oceanographic research.",
    },
    "prod_tidewatch": {
        "id": "prod_tidewatch", "type": "product",
        "name": "TideWatch Monitoring System", "manufacturer_id": "comp_oceanic",
        "launched_year": 2022, "price": 45000,
        "awards_won": [],
        "description": "A coastal monitoring system for tracking tidal patterns and erosion.",
    },
    "prod_voltcell": {
        "id": "prod_voltcell", "type": "product",
        "name": "VoltCell Battery Pack", "manufacturer_id": "comp_terravolt",
        "launched_year": 2021, "price": 8500,
        "awards_won": [],
        "description": "A high-capacity solid-state battery pack for residential solar storage.",
    },
    "prod_gridbox": {
        "id": "prod_gridbox", "type": "product",
        "name": "GridBox Power Controller", "manufacturer_id": "comp_terravolt",
        "launched_year": 2023, "price": 4200,
        "awards_won": [],
        "description": "A smart grid controller for managing distributed energy resources.",
    },
    "prod_optinet": {
        "id": "prod_optinet", "type": "product",
        "name": "OptiNet Optimizer", "manufacturer_id": "comp_vertexai",
        "launched_year": 2021, "price": 75000,
        "awards_won": [],
        "description": "A network optimization tool powered by reinforcement learning.",
    },
    "prod_skyseed": {
        "id": "prod_skyseed", "type": "product",
        "name": "SkySeed Agricultural Drone", "manufacturer_id": "comp_canopy",
        "launched_year": 2022, "price": 12000,
        "awards_won": ["award_atlas"],
        "description": "An autonomous drone for crop monitoring and precision seeding.",
    },
    "prod_wayfinder": {
        "id": "prod_wayfinder", "type": "product",
        "name": "Wayfinder Navigation Module", "manufacturer_id": "comp_polaris",
        "launched_year": 2015, "price": 6800,
        "awards_won": [],
        "description": "A high-precision GPS/INS navigation module for autonomous vehicles.",
    },
    "prod_frostcore": {
        "id": "prod_frostcore", "type": "product",
        "name": "FrostCore Cooling System", "manufacturer_id": "comp_arctic",
        "launched_year": 2023, "price": 95000,
        "awards_won": [],
        "description": "An immersion cooling system for data centers using biodegradable coolant.",
    },
}

# Unified KB — all entities by ID
KB = {}
KB.update(CITIES)
KB.update(UNIVERSITIES)
KB.update(AWARDS)
KB.update(PEOPLE)
KB.update(COMPANIES)
KB.update(PRODUCTS)


# ── Tool Functions ──────────────────────────────────────────────

def search_kb(query):
    """Fuzzy search across all entity names and descriptions. Returns top 3 matches as summaries with IDs."""
    query_lower = query.lower()
    query_words = set(query_lower.split())

    scored = []
    for eid, entity in KB.items():
        name = entity.get("name", "").lower()
        desc = entity.get("description", "").lower()
        etype = entity.get("type", "")
        searchable = f"{name} {desc}"

        # Score: exact name match > word overlap > substring
        score = 0
        if query_lower == name:
            score = 100
        elif query_lower in name:
            score = 80
        elif name in query_lower:
            score = 70
        else:
            # Word overlap scoring
            searchable_words = set(searchable.split())
            overlap = query_words & searchable_words
            score = len(overlap) * 10

            # Substring bonus
            for word in query_words:
                if len(word) > 2 and word in searchable:
                    score += 5

        if score > 0:
            summary = f"[{eid}] ({etype}) {entity['name']}: {entity.get('description', '')[:120]}"
            scored.append((score, summary))

    scored.sort(key=lambda x: -x[0])
    results = [s[1] for s in scored[:3]]
    if not results:
        return "No results found."
    return "\n".join(results)


def lookup_kb(entity_id):
    """Look up full details for an entity by its ID. Returns JSON string."""
    entity_id = entity_id.strip()
    entity = KB.get(entity_id)
    if entity is None:
        # Try fuzzy ID match (strip whitespace, lowercase)
        for eid, ent in KB.items():
            if eid.lower() == entity_id.lower():
                entity = ent
                break
    if entity is None:
        return f"Entity '{entity_id}' not found. Use search to find the correct ID."
    return json.dumps(entity, indent=2)


def calculate(expression):
    """Safely evaluate a mathematical expression. Returns the result as a string."""
    try:
        # Allow basic math operations and functions
        allowed_names = {
            "abs": abs, "round": round, "min": min, "max": max,
            "sum": sum, "len": len, "int": int, "float": float,
            "pow": pow, "sqrt": math.sqrt, "ceil": math.ceil, "floor": math.floor,
        }
        # Remove anything that's not a number, operator, or allowed function
        clean = expression.strip()
        result = eval(clean, {"__builtins__": {}}, allowed_names)
        return str(result)
    except Exception as e:
        return f"Error evaluating '{expression}': {e}"


def patch_agent_tools(mod):
    """Monkey-patch a generated agent module's tool_* functions with KB implementations.

    Matches by keyword in function name:
      - search → search_kb
      - lookup → lookup_kb
      - calcul → calculate
    """
    patched = []
    for attr_name in dir(mod):
        if not attr_name.startswith("tool_"):
            continue
        name_lower = attr_name.lower()
        if "search" in name_lower:
            setattr(mod, attr_name, search_kb)
            patched.append(f"{attr_name} → search_kb")
        elif "lookup" in name_lower:
            setattr(mod, attr_name, lookup_kb)
            patched.append(f"{attr_name} → lookup_kb")
        elif "calcul" in name_lower:
            setattr(mod, attr_name, calculate)
            patched.append(f"{attr_name} → calculate")
    return patched

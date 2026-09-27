import requests
import numpy as np
import math
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==============================================================================
# ENVIRONMENT ROUTING CONFIGURATION
# ==============================================================================
API_SPORTS_KEY = "38f89f91868c570c376725e8cec3ed15"
BASE_URL = "https://v3.football.api-sports.io"
HEADERS = {"x-apisports-key": API_SPORTS_KEY}

TEAM_DATABASE_MAPPING = {
    "nigeria": 29, "senegal": 22, "egypt": 30, "morocco": 31, "cameroon": 1530,
    "jordan": 1104, "syria": 1103, "ghana": 1504, "ivory coast": 1166, 
    "south africa": 1172, "tunisia": 28,
    "algeria": 32, "niger": 1167, "lesotho": 1152, "mozambique": 1174, "guinea": 24,
    "france": 2, "england": 10, "italy": 768, "belgium": 764, "germany": 25,
    "spain": 9, "portugal": 27, "netherlands": 1118, "croatia": 3,
    "man united": 33, "man city": 50, "chelsea": 49, "arsenal": 42, "liverpool": 40,
    "real madrid": 541, "barcelona": 529, "juventus": 496, "ac milan": 489, "psg": 85
}

class MatchRequest(BaseModel):
    home_team: str
    away_team: str

# ==============================================================================
# STATISTICAL POISSON MATHEMATICS
# ==============================================================================
def calculate_poisson_probability(lambda_val, k):
    return (math.exp(-lambda_val) * (lambda_val ** k)) / math.factorial(k)

def calculate_under_probability(lambda_val, line):
    prob_sum = 0.0
    for k in range(0, math.floor(line) + 1):
        prob_sum += calculate_poisson_probability(lambda_val, k)
    return prob_sum * 100

# ==============================================================================
# SECTOR 1: MULTI-SEASON AGGREGATION LOOP (LAST 5 SEASONS CRUNCHER)
# ==============================================================================
def compile_five_season_player_metrics(team_id):
    """Iterates loops over the last 5 years to compile squad offensive/defensive form weights."""
    attacker_scores = []
    defender_scores = []
    
    # Dynamically scan the last 5 completed seasonal timelines
    for target_year in [2021, 2022, 2023, 2024, 2025]:
        url = f"{BASE_URL}/players?team={team_id}&season={target_year}"
        try:
            response = requests.get(url, headers=HEADERS)
            if response.status_code != 200: continue
            player_data = response.json().get("response", [])
            if not player_data: continue
            
            for item in player_data:
                stats_list = item.get("statistics", [{}])
                if not stats_list: continue
                stats = stats_list[0]
                
                games = stats.get("games", {}).get("appearences", 0) or 1
                position = stats.get("games", {}).get("position", "Attacker")
                goals = stats.get("goals", {}).get("total", 0) or 0
                assists = stats.get("goals", {}).get("assists", 0) or 0
                tackles = stats.get("tackles", {}).get("total", 0) or 0
                
                if position in ["Attacker", "Midfielder"]:
                    attacker_scores.append(((goals * 15) + (assists * 7)) / games)
                elif position in ["Defender", "Goalkeeper"]:
                    defender_scores.append((tackles * 5) / games)
        except:
            continue
            
    top_attack = sorted(attacker_scores, reverse=True)[:5]
    top_defense = sorted(defender_scores, reverse=True)[:5]
    
    return {
        "attack_index": np.mean(top_attack) if top_attack else 50.0,
        "defense_index": np.mean(top_defense) if top_defense else 50.0
    }

# ==============================================================================
# SECTOR 2: FIRES 10-MATCH OVERALL fixtures LEDGER FILTER
# ==============================================================================
def analyze_last_10_fixtures_overall(team_id):
    """Pulls exactly the last 10 completed matches against any opponent to map form trends."""
    url = f"{BASE_URL}/fixtures?team={team_id}&last=10&status=FT"
    try:
        response = requests.get(url, headers=HEADERS)
        if response.status_code != 200: return {"avg_scored": 1.5, "avg_conceded": 1.2, "gg_rate": 50.0}
        fixtures = response.json().get("response", [])
        if not fixtures: return {"avg_scored": 1.5, "avg_conceded": 1.2, "gg_rate": 50.0}
        
        goals_scored, goals_conceded, gg_count = [], [], 0
        for fix in fixtures:
            goals = fix.get("goals", {})
            h_g = goals.get("home", 0) or 0
            a_g = goals.get("away", 0) or 0
            is_home = fix["teams"]["home"]["id"] == team_id
            
            goals_scored.append(h_g if is_home else a_g)
            goals_conceded.append(a_g if is_home else h_g)
            if h_g > 0 and a_g > 0: gg_count += 1
            
        return {
            "avg_scored": np.mean(goals_scored) if goals_scored else 1.5,
            "avg_conceded": np.mean(goals_conceded) if goals_conceded else 1.2,
            "gg_rate": (gg_count / len(fixtures)) * 100
        }
    except:
        return {"avg_scored": 1.5, "avg_conceded": 1.2, "gg_rate": 50.0}

# ==============================================================================
# SECTOR 3: HEAD-TO-HEAD DIRECT MATCHING LAYER (HISTORICAL H2H CLASHES)
# ==============================================================================
def compile_direct_h2h_modifier(home_id, away_id):
    """Queries direct history matchups between these two teams to weight structural bias."""
    url = f"{BASE_URL}/fixtures/headtohead?h2h={home_id}-{away_id}"
    try:
        response = requests.get(url, headers=HEADERS)
        if response.status_code != 200: return {"home_bias": 1.0, "away_bias": 1.0, "h2h_goals": 2.5}
        fixtures = response.json().get("response", [])
        if not fixtures: return {"home_bias": 1.0, "away_bias": 1.0, "h2h_goals": 2.5}
        
        home_wins, away_wins, total_goals = 0, 0, []
        for fix in fixtures:
            goals = fix.get("goals", {})
            h_g = goals.get("home", 0) or 0
            a_g = goals.get("away", 0) or 0
            total_goals.append(h_g + a_g)
            
            if fix["teams"]["home"]["id"] == home_id:
                if h_g > a_g: home_wins += 1
                elif a_g > h_g: away_wins += 1
            else:
                if a_g > h_g: home_wins += 1
                elif h_g > a_g: away_wins += 1
                
        total_fixtures = len(fixtures)
        return {
            "home_bias": 1.0 + (home_wins / total_fixtures),
            "away_bias": 1.0 + (away_wins / total_fixtures),
            "h2h_goals": np.mean(total_goals) if total_goals else 2.5
        }
    except:
        return {"home_bias": 1.0, "away_bias": 1.0, "h2h_goals": 2.5}

# ==============================================================================
# COMPILER ENDPOINT
# ==============================================================================
@app.post("/predict")
def predict_match(request: MatchRequest):
    home_name = request.home_team.strip().lower()
    away_name = request.away_team.strip().lower()
    
    home_id = TEAM_DATABASE_MAPPING.get(home_name, 49)
    away_id = TEAM_DATABASE_MAPPING.get(away_name, 42)

    # 1. Gather 5-Season Data Matrices
    home_squad = compile_five_season_player_metrics(home_id)
    away_squad = compile_five_season_player_metrics(away_id)
    
    # 2. Extract 10-Fixtures Overall Stats
    home_history = analyze_last_10_fixtures_overall(home_id)
    away_history = analyze_last_10_fixtures_overall(away_id)
    
    # 3. Compile Direct H2H History Modifiers
    h2h_data = compile_direct_h2h_modifier(home_id, away_id)

    # Advanced Multi-Vector Synthesis Matrix Formula mapping
    home_final = ((home_history["avg_scored"] * home_squad["attack_index"]) + 
                  (home_history["avg_conceded"] / (away_squad["defense_index"] or 1))) * h2h_data["home_bias"]
                  
    away_final = ((away_history["avg_scored"] * away_squad["attack_index"]) + 
                  (away_history["avg_conceded"] / (home_squad["defense_index"] or 1))) * h2h_data["away_bias"]
                  
    draw_final = (home_final + away_final) * 0.22

    total_matrix = home_final + away_final + draw_final
    home_win_pct = (home_final / total_matrix) * 100
    away_win_pct = (away_final / total_matrix) * 100
    draw_pct = (draw_final / total_matrix) * 100
    
    # Blend general averages with exact historical matching metrics lines
    avg_goals = ((home_history["avg_scored"] + away_history["avg_scored"]) * 0.7) + (h2h_data["h2h_goals"] * 0.3)
    gg_prob = (home_history["gg_rate"] + away_history["gg_rate"]) / 2

    markets_comparison = {
        f"{home_name.upper()} WIN (1)": home_win_pct,
        f"{away_name.upper()} WIN (2)": away_win_pct,
        "HOME OR DRAW (1X)": home_win_pct + draw_pct,
        "AWAY OR DRAW (X2)": away_win_pct + draw_pct,
        "HOME OR AWAY (12)": home_win_pct + away_win_pct,
        "STRAIGHT DRAW (X)": draw_pct,
        "GOAL GOAL (GG / BTTS)": gg_prob,
        "NO GOAL (NG)": 100 - gg_prob
    }

    for line in [0.5, 1.5, 2.5, 3.5, 4.5]:
        under_prob = calculate_under_probability(avg_goals, line)
        markets_comparison[f"MATCH OVER {line} GOALS"] = 100.0 - under_prob
        markets_comparison[f"MATCH UNDER {line} GOALS"] = under_prob

    sorted_markets = sorted(markets_comparison.items(), key=lambda x: x[1], reverse=True)
    
    option_1_name, option_1_prob = sorted_markets[0]
    option_2_name, option_2_prob = sorted_markets[1]
    
    return {
        "home_team": home_name.upper(),
        "away_team": away_name.upper(),
        "primary_option": option_1_name,
        "primary_confidence": f"{option_1_prob:.2f}%",
        "secondary_option": option_2_name,
        "secondary_confidence": f"{option_2_prob:.2f}%"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8090)

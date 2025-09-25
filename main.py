import pandas as pd
import glob
import os
import re
from itertools import permutations
import random

def load_provider_results_in_dfs(path):
    result_files = glob.glob(path)
    provider_results = {}

    for file_path in result_files:
        base = os.path.basename(file_path)
        provider = base.split('-')[0].upper()
        df_var_name = f"{provider.lower()}_df"
        globals()[df_var_name] = pd.read_csv(file_path)
        provider_results[provider] = globals()[df_var_name]
    return provider_results

def normalize_provider_df_final(df: pd.DataFrame, provider_name: str) -> pd.DataFrame:
    df.columns = df.columns.str.strip()

    # Fix PropertyID
    if "PropertyID" not in df.columns:
        id_candidates = [col for col in df.columns if "id" in col.lower()]
        if not id_candidates:
            raise ValueError(f"{provider_name} missing a valid ID column.")
        df = df.rename(columns={id_candidates[0]: "PropertyID"})

    # Ensure address and zip columns match leads
    required_columns = ["PropertyAddress", "PropertyZip"]
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"{provider_name} missing required column '{col}'.")

    # Normalize phone columns using regex
    phone_col_map = {}
    for col in df.columns:
        match = re.search(r'\b(?:p?hone)?\s*number\s*(\d+)', col, flags=re.IGNORECASE)
        if match:
            num = int(match.group(1))
            phone_col_map[col] = f"Phone number {num}"

    df = df.rename(columns=phone_col_map)

    # Ensure at least one valid phone column
    if not any(col.startswith("Phone number") for col in df.columns):
        raise ValueError(f"{provider_name} has no valid phone columns.")

    df["PropertyID"] = pd.to_numeric(df["PropertyID"], errors="coerce").astype("Int64")

    # Drop rows where PropertyID is <NA>
    df = df.dropna(subset=["PropertyID"])

    # Check for duplicates in PropertyID
    duplicate_ids = df[df.duplicated("PropertyID", keep=False)]["PropertyID"].unique()
    if len(duplicate_ids) > 0:
        raise ValueError(f"{provider_name} has duplicate PropertyIDs after cleaning: {duplicate_ids.tolist()}")

    return df

def build_contactability_matrix(normalized_results: dict) -> pd.DataFrame:
    contact_matrix = {}

    for provider, df in normalized_results.items():
        # Get phone number columns
        phone_cols = [col for col in df.columns if col.startswith("Phone number")]
        # A lead is contacted if any phone field is non-null
        contact_series = df[phone_cols].notnull().any(axis=1).astype(int)
        # Map to PropertyID
        contact_matrix[provider] = pd.Series(contact_series.values, index=df["PropertyID"])

    # Combine into a full matrix
    contact_df = pd.DataFrame(contact_matrix).fillna(0).astype(int)
    contact_df.index.name = "PropertyID"
    return contact_df.sort_index()

def evaluate_sequence(sequence, contact_matrix, cost_dict):
    seen_ids = set()
    total_contacts = 0
    total_cost = 0
    step_log = []

    for provider in sequence:
        ids = contact_matrix.index[contact_matrix[provider] == 1].difference(seen_ids)
        new_contacts = len(ids)
        cost = cost_dict[provider]

        total_contacts += new_contacts
        total_cost += cost * new_contacts
        seen_ids.update(ids)

        step_log.append({
            "provider": provider,
            "new_contacts": new_contacts,
            "cumulative_contacts": total_contacts,
            "cumulative_cost": total_cost
        })

    return {
        "sequence": sequence,
        "total_contacts": total_contacts,
        "total_cost": total_cost,
        "cost_per_contact": total_cost / total_contacts if total_contacts else float("inf"),
        "steps": step_log
    }

def brute_force_with_log(contact_matrix, cost_dict):
    provider_list = list(contact_matrix.columns)
    all_results = {}

    for seq in permutations(provider_list):
        result = evaluate_sequence(seq, contact_matrix, cost_dict)
        all_results[seq] = result

    best_seq = min(all_results.items(), key=lambda x: x[1]["cost_per_contact"])

    return all_results, best_seq

def heuristic_with_log(contact_matrix, cost_dict):
    remaining_providers = set(contact_matrix.columns)
    seen_ids = set()
    total_contacts = 0
    total_cost = 0
    sequence = []
    step_log = []

    while remaining_providers:
        best_provider = None
        best_efficiency = 0
        best_new_contacts = 0

        for provider in remaining_providers:
            ids = contact_matrix.index[contact_matrix[provider] == 1].difference(seen_ids)
            new_contacts = len(ids)
            cost = cost_dict[provider]

            if new_contacts > 0:
                efficiency = 1 / cost  # Pure cost-based ranking since all new contacts count equally
                if efficiency > best_efficiency:
                    best_efficiency = efficiency
                    best_provider = provider
                    best_new_contacts = new_contacts

        if best_provider is None:
            break

        cost = cost_dict[best_provider]
        total_contacts += best_new_contacts
        total_cost += cost * best_new_contacts
        seen_ids.update(contact_matrix.index[contact_matrix[best_provider] == 1])
        remaining_providers.remove(best_provider)
        sequence.append(best_provider)

        step_log.append({
            "provider": best_provider,
            "new_contacts": best_new_contacts,
            "cumulative_contacts": total_contacts,
            "cumulative_cost": total_cost
        })

    return {
        "sequence": sequence,
        "total_contacts": total_contacts,
        "total_cost": total_cost,
        "cost_per_contact": total_cost / total_contacts if total_contacts else float("inf"),
        "steps": step_log
    }

def genetic_algorithm_contact_order_with_log(matrix, vendor_costs, population_size=20, generations=50, mutation_rate=0.1):
    providers = list(matrix.columns)

    def fitness(seq):
        result = evaluate_sequence(seq, matrix, vendor_costs)
        return -result['cost_per_contact'], result

    def mutate(seq):
        a, b = random.sample(range(len(seq)), 2)
        seq = list(seq)
        seq[a], seq[b] = seq[b], seq[a]
        return tuple(seq)

    population = [tuple(random.sample(providers, len(providers))) for _ in range(population_size)]
    generation_log = []

    for gen in range(generations):
        scored = [(fitness(seq)[0], seq) for seq in population]
        top_half = [seq for _, seq in sorted(scored, reverse=True)[:population_size // 2]]

        best_cost = -max(scored)[0]
        avg_cost = -sum(score for score, _ in scored) / len(scored)
        best_seq = max(scored)[1]
        generation_log.append({
            "generation": gen,
            "best_cost_per_contact": best_cost,
            "avg_cost_per_contact": avg_cost,
            "best_sequence": " > ".join(best_seq)
        })

        children = []
        for _ in range(population_size - len(top_half)):
            parent1, parent2 = random.sample(top_half, 2)
            cut = random.randint(1, len(providers) - 2)
            child = list(parent1[:cut]) + [p for p in parent2 if p not in parent1[:cut]]
            if random.random() < mutation_rate:
                child = mutate(child)
            children.append(tuple(child))

        population = top_half + children

    # Final evaluation of the best sequence
    final_best_seq = max(population, key=lambda seq: fitness(seq)[0])
    final_result = evaluate_sequence(final_best_seq, matrix, vendor_costs)

    return final_result, pd.DataFrame(generation_log)


if __name__ == "__main__":
    provider_results = load_provider_results_in_dfs('./data/t*-results.csv')

    #print({provider: df.columns.tolist() for provider, df in provider_results.items()})

    leads = pd.read_csv('./data/leads-2-.csv')

    # Apply the refined normalization function
    normalized_results_refined = {
        k: normalize_provider_df_final(df, k) for k, df in provider_results.items()
    }

    # Confirm the results again
    #print({provider: df.columns.tolist() for provider, df in normalized_results_refined.items()})

    matrix = build_contactability_matrix(normalized_results_refined)
    print(matrix.head())
    # from the pdf
    vendor_cost_per_hit = {
        "T1": 0.08,
        "T2": 0.06,
        "T3": 0.005,
        "T4": 0.005,
        "T5": 0.02,
        "T6": 0.06
    }

    #brute_force_result_all, brute_force_best = brute_force_with_log(matrix, vendor_cost_per_hit)
    #heuristic_result = heuristic_with_log(matrix, vendor_cost_per_hit)
    #genetic_result, genetic_log = genetic_algorithm_contact_order_with_log(matrix, vendor_cost_per_hit)
#
    #print("Brute Force Results:")
    #print(brute_force_best)
    #print("\nHeuristic Results:")
    #print(heuristic_result)
    #print("\nGenetic Algorithm Results:")
    #print(genetic_result)

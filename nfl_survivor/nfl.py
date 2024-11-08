def elo_update(team1, team2, result, power_ranking, k=20):
    """
    Updates the Elo ratings of two teams after a game.

    Parameters:
        team1 (str): Name of team 1.
        team2 (str): Name of team 2.
        result (int): Indicates the game outcome: 1 if team1 wins, 2 if team2 wins, 0 if it's a tie.
        power_ranking (pd.DataFrame): DataFrame containing the teams and their Elo ratings.
        k (float): The K-factor determining the sensitivity of Elo adjustments. Default is 20.

    Returns:
        tuple: Updated Elo ratings (new_team1_elo, new_team2_elo).
    """
    # Get Elo ratings from the DataFrame for the teams
    team1_elo = power_ranking.loc[power_ranking['Team'] == team1, 'ELO-score'].values[0]
    team2_elo = power_ranking.loc[power_ranking['Team'] == team2, 'ELO-score'].values[0]

    # Calculate the expected outcome for each team
    expected_team1 = 1 / (1 + 10 ** ((team2_elo - team1_elo) / 400))
    expected_team2 = 1 - expected_team1

    # Set the actual scores based on the result
    if result == 1:
        score_team1, score_team2 = 1, 0  # Team 1 wins
    elif result == 2:
        score_team1, score_team2 = 0, 1  # Team 2 wins
    elif result == 0:
        score_team1, score_team2 = 0.5, 0.5  # Tie
    else:
        raise ValueError("Result must be 1 (team1 wins), 2 (team2 wins), or 0 (tie)")

    # Calculate the new Elo ratings
    new_team1_elo = team1_elo + k * (score_team1 - expected_team1)
    new_team2_elo = team2_elo + k * (score_team2 - expected_team2)

    # Update the DataFrame with the new Elo ratings
    power_ranking.loc[power_ranking['Team'] == team1, 'ELO-score'] = new_team1_elo
    power_ranking.loc[power_ranking['Team'] == team2, 'ELO-score'] = new_team2_elo

    return power_ranking

def win_probability(team1_elo, team2_elo, team1_home=True, home_field_advantage=0):
    """
    Calculates the probability of team 1 winning against team 2 given their Elo ratings and home advantage.

    Parameters:
        team1_elo (float): Elo rating of team 1.
        team2_elo (float): Elo rating of team 2.
        team1_home (bool): Whether team 1 is playing at home. Default is True.
        home_field_advantage (float): Elo point adjustment for the home team. Default is 65.

    Returns:
        float: Probability of team 1 winning against team 2.
    """
    # Adjust for home-field advantage
    if team1_home:
        adjusted_team1_elo = team1_elo + home_field_advantage
        adjusted_team2_elo = team2_elo
    else:
        adjusted_team1_elo = team1_elo
        adjusted_team2_elo = team2_elo + home_field_advantage

    # Calculate the win probability for team 1
    win_prob_team1 = 1 / (1 + 10 ** ((adjusted_team2_elo - adjusted_team1_elo) / 400))

    return win_prob_team1

def power_ranking_update(team, new_elo, power_ranking):
    """
    Updates the Elo score of a specified team in the power_ranking DataFrame.

    Parameters:
        team (str): The name of the team whose Elo score should be updated.
        new_elo (float): The new Elo score for the team.
        power_ranking (pd.DataFrame): The DataFrame containing team power rankings with 'Team' and 'ELO-score' columns.

    Returns:
        pd.DataFrame: The updated power_ranking DataFrame.
    """
    # Check if the team exists in the DataFrame
    if team in power_ranking['Team'].values:
        # Update the ELO-score for the specified team
        power_ranking.loc[power_ranking['Team'] == team, 'ELO-score'] = new_elo
        print(f"Updated {team}'s ELO-score to {new_elo}.")
    else:
        print(f"Team '{team}' not found in the power ranking DataFrame.")

    return power_ranking

def weekly_update(matches, results):
    for match, result in zip(matches, results):
        team1, team2 = match.split('-')
        goals1, goals2 = result.split('-')
        if goals1 > goals2:
            elo_update(team1, team2, 1, power_ranking)
        elif goals1 == goals2:
            elo_update(team1, team2, 0, power_ranking)
        elif goals1 < goals2:
            elo_update(team1, team2, 1, power_ranking)
    return None

def get_matches_with_probabilities(week):
    """
    Get the matches with probabilities for a specific week, using Elo ratings and home field advantage.

    Parameters:
        schedule (DataFrame): DataFrame containing matchups, with teams as rows and opponent teams as columns.
        power_ranking (DataFrame): DataFrame containing the Elo ratings of teams.
        week (int): The week number to filter matches by (e.g., 1 for week 1).
        home_field_advantage (float): The home-field advantage bonus (default 65).

    Returns:
        DataFrame: A table containing teams, their opponents, and the match probabilities.
    """
    # Extract the relevant column for the given week (assuming columns are named like 'week 1', 'week 2', ...)
    week_column = f"Wk{week}"

    # Get the teams and opponents from the schedule for the given week
    teams = schedule.iloc[:, 0]
    opponents = schedule[week_column]

    # Create a new DataFrame with the teams and their opponents
    matches_df = pd.DataFrame({
        'Team': teams,
        'Opponent': opponents
    })

    # Calculate the Elo ratings for the teams
    elo_ratings = {team: power_ranking.loc[power_ranking['Team'] == team, 'ELO-score'].values[0] 
                   for team in teams}

    # Calculate the probabilities for each match using the win_probability function
    probabilities = []
    for index, row in matches_df.iterrows():
        team1 = row['Team']
        team2 = row['Opponent']

        # Get the Elo ratings for the teams
        team1_elo = elo_ratings.get(team1)
        team2_elo = elo_ratings.get(team2)

        if team1_elo is not None and team2_elo is not None:
            # Calculate the probability using the win_probability function
            probability = win_probability(team1_elo, team2_elo, team1_home=True)
            probabilities.append(probability)
        else:
            probabilities.append(float('-inf'))  # Handle bye matches

    # Add the probabilities to the DataFrame
    matches_df['Probability'] = probabilities

    return matches_df

#import files from computer
import pandas as pd

# Define the folder path
folder_path = 'NFL/'

# Read each CSV file
#match_results = pd.read_csv(f'{folder_path}match_results.csv', delimiter=';')
match_results = pd.read_csv(r'C:\Users\mpfle\Documents\opkut3\nfl_survivor\NFL\match_results.csv', delimiter=';')

#power_ranking = pd.read_csv(f'{folder_path}power_ranking.csv', delimiter=';')
power_ranking = pd.read_csv(r'C:\Users\mpfle\Documents\opkut3\nfl_survivor\NFL\power_ranking.csv', delimiter=';')
power_ranking['ELO-score'] = power_ranking['ELO-score'].astype(float)

#schedule = pd.read_csv(f'{folder_path}schedule.csv', delimiter=';')
schedule = pd.read_csv(r'C:\Users\mpfle\Documents\opkut3\nfl_survivor\NFL\schedule.csv', delimiter=';')

winners = ['BUF','TB']

#updating the ratings according to the results of the first two weeks
weekly_update(match_results["week 1 matching"], match_results["week 1 results"])
weekly_update(match_results["week 2 matching"], match_results["week 2 results"])

#calculating the probabilities of each team winning in each round
week_tables = []
for week in range(3, 19):  # 19 is exclusive, so this will go from 3 to 18
    week_data = get_matches_with_probabilities(week)  # Get matches with probabilities for the week
    week_data = week_data[week_data["Probability"] != float('-inf')]
    globals()[f"week_{week}"] = week_data.sort_values(by="Probability", ascending=False)  # Dynamically create variables like week_3, week_4, ...
    week_tables.append(globals()[f"week_{week}"])



#solution 1 (greedy algorithm)


for week_table in week_tables:
    for _, row in week_table.iterrows():
        team = row["Team"]
        if team not in winners:
            winners.append(team)
            break

winners_greedy = winners


#Solution by ChatGPT (full lookahead)

winners = ['BUF','TB']

from pulp import *
import numpy as np

# Define the optimization problem
survivor = LpProblem("survivor", LpMaximize)

# Define the binary decision variables
x = LpVariable.dicts(
    'x', 
    [(week, team) for week, week_table in enumerate(week_tables) for team in week_table["Team"]],
    cat="Binary"
)

# Objective function: maximize the sum of winning probabilities, filtering out inf or NaN
survivor += lpSum(
    x[(week, team)] * np.log(
        week_table.loc[week_table["Team"] == team, "Probability"]
        .replace([np.inf, -np.inf], np.nan)  # Handle any infinite probabilities
        .fillna(0)  # Convert NaN probabilities to 0
        .values[0]
    )
    for week, week_table in enumerate(week_tables)
    for team in week_table["Team"]
    if not pd.isnull(week_table.loc[week_table["Team"] == team, "Probability"]
                     .replace([np.inf, -np.inf], np.nan)
                     .values[0])  # Exclude teams with invalid probabilities
), "Maximize_Product_of_Probabilities"

# Add a constraint to ensure 'BUF' and 'TB' cannot be selected
for team in winners:
    for week, week_table in enumerate(week_tables):
        if team in week_table["Team"]:
            survivor += x[(week, team)] == 0, f"Exclude_{team}_from_week_{week}"

# Constraint 1: Only one team per week
for week, week_table in enumerate(week_tables):
    survivor += lpSum(x[(week, team)] for team in week_table["Team"]) == 1, f"One_team_per_week_{week}"

# Constraint 2: Each team can only be selected once
all_teams = set(team for week_table in week_tables for team in week_table["Team"])
for team in all_teams:
    survivor += lpSum(x[(week, team)] for week, week_table in enumerate(week_tables) if team in week_table["Team"]) <= 1, f"One_time_team_{team}"

# Solve the problem
survivor.solve()

# Loop through each week and append the chosen team to the winners list
for week, week_table in enumerate(week_tables):
    for team in week_table["Team"]:
        if x[(week, team)].varValue == 1:  # If the team is selected (binary decision variable is 1)
            winners.append(team)  # Add to winners list
            #print(f"Week {week + 1}: {team} is chosen")  # Print the team chosen for that week

winners_full_lookahead = winners


#Define problem with look_ahead = 4 (aim is to maximize the probability of surviving for 4 weeks, than a greedy algorthm)

look_ahaed = 5
winners = ['BUF','TB']

# Define the optimization problem
survivor = LpProblem("survivor", LpMaximize)

# Define the binary decision variables
x = LpVariable.dicts(
    'x', 
    [(week, team) for week, week_table in enumerate(week_tables[:look_ahaed]) for team in week_table["Team"]],
    cat="Binary"
)

# Objective function: maximize the sum of winning probabilities, filtering out inf or NaN
survivor += lpSum(
    x[(week, team)] * np.log(
        week_table.loc[week_table["Team"] == team, "Probability"]
        .replace([np.inf, -np.inf], np.nan)  # Handle any infinite probabilities
        .fillna(0)  # Convert NaN probabilities to 0
        .values[0]
    )
    for week, week_table in enumerate(week_tables[:look_ahaed])
    for team in week_table["Team"]
    if not pd.isnull(week_table.loc[week_table["Team"] == team, "Probability"]
                     .replace([np.inf, -np.inf], np.nan)
                     .values[0])  # Exclude teams with invalid probabilities
), "Maximize_Product_of_Probabilities"

# Add a constraint to ensure 'BUF' and 'TB' cannot be selected
for team in winners:
    for week, week_table in enumerate(week_tables[:look_ahaed]):
        if team in week_table["Team"]:
            survivor += x[(week, team)] == 0, f"Exclude_{team}_from_week_{week}"

# Constraint 1: Only one team per week
for week, week_table in enumerate(week_tables[:look_ahaed]):
    survivor += lpSum(x[(week, team)] for team in week_table["Team"]) == 1, f"One_team_per_week_{week}"

# Constraint 2: Each team can only be selected once
all_teams = set(team for week_table in week_tables for team in week_table["Team"])
for team in all_teams:
    survivor += lpSum(x[(week, team)] for week, week_table in enumerate(week_tables[:look_ahaed]) if team in week_table["Team"]) <= 1, f"One_time_team_{team}"

# Solve the problem
survivor.solve()

# Loop through each week and append the chosen team to the winners list
for week, week_table in enumerate(week_tables[:look_ahaed]):
    for team in week_table["Team"]:
        if x[(week, team)].varValue == 1:  # If the team is selected (binary decision variable is 1)
            winners.append(team)  # Add to winners list
            #print(f"Week {week + 1}: {team} is chosen")  # Print the team chosen for that week

for week_table in week_tables[look_ahaed:]:
    for _, row in week_table.iterrows():
        team = row["Team"]
        if team not in winners:
            winners.append(team)
            break

winners_with_lookahead = winners


# Print the final list of winners
print("Final choice with greedy algorithm: \n", winners_greedy)
print("Final choice with full lookahead: \n", winners_full_lookahead)
print("Final choice with lookahead: \n", winners_with_lookahead)




#comparison of performance

# Define a function to calculate the cumulative product of probabilities for a given winners vector
def calculate_cumulative_probability(winners_vector, week_tables, n_weeks):
    cumulative_prob = 1
    winners_vector = winners_vector[2:n_weeks+2]
    for week, week_table in enumerate(week_tables[:n_weeks]):  # Adjust to use only weeks up to n_weeks
        team = winners_vector[week]  # Get the team chosen in this week
        # Get the probability of the team for the current week
        team_probability = week_table.loc[week_table["Team"] == team, "Probability"].replace([np.inf, -np.inf], np.nan).fillna(0).values[0]
        cumulative_prob *= team_probability  # Multiply the probabilities
    return cumulative_prob

# Define the nth week you want to compare up to
n_weeks = 10

# Calculate cumulative probabilities for each winner vector (up to week n)
prob_greedy = calculate_cumulative_probability(winners_greedy, week_tables, n_weeks)
prob_full_lookahead = calculate_cumulative_probability(winners_full_lookahead, week_tables, n_weeks)
prob_with_lookahead = calculate_cumulative_probability(winners_with_lookahead, week_tables, n_weeks)

# Print or return the results for comparison
print(f"Performance for Greedy (up to week {n_weeks}): {prob_greedy:.4f}")
print(f"Performance for Full Lookahead (up to week {n_weeks}): {prob_full_lookahead:.4f}")
print(f"Performance for Lookahead (up to week {n_weeks}): {prob_with_lookahead:.4f}")
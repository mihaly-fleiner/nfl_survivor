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
            probabilities.append(None)  # Handle missing Elo data

    # Add the probabilities to the DataFrame
    matches_df['Probability'] = probabilities

    return matches_df


"""#round 1
for match, result in zip(match_results["week 1 matching"], match_results["week 1 results"]):
    team1, team2 = match.split('-')
    goals1, goals2 = result.split('-')
    if goals1 > goals2:
        elo_update(team1, team2, 1, power_ranking)
    elif goals1 == goals2:
        elo_update(team1, team2, 0, power_ranking)
    elif goals1 < goals2:
        elo_update(team1, team2, 1, power_ranking)"""

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

"""# Display the first few rows of each DataFrame to verify
print("Match Results:")
print(match_results.head(), "\n")

print("Power Ranking:")
print(power_ranking.head(), "\n")

print("Schedule:")
print(schedule.head())"""

#print(power_ranking.head())
weekly_update(match_results["week 1 matching"], match_results["week 1 results"])
weekly_update(match_results["week 2 matching"], match_results["week 2 results"])
#print(power_ranking.head())

week = 3
week_3 = get_matches_with_probabilities(week)
print(week_3.head())
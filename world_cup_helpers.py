from collections import defaultdict
from time import strftime
import pandas as pd
import numpy as np
from keras import backend, Sequential, regularizers, optimizers
from keras.models import Model
from keras.layers import Input, Masking, LSTM, Dense, concatenate
from keras.callbacks import TensorBoard

GOAL_DECIMALS = 2
WIN_FACTOR = 1.5

# Read country mapping
country_mapping = pd.read_csv('./country_mapping.txt', header=None, index_col=0).T.iloc[0].to_dict()

# Read experience data and map country names
experience_df = pd.read_csv('./experience.csv')
experience_df.loc[:, 'cnty'] = experience_df['cnty'].apply(lambda x: country_mapping[x] if x in country_mapping else x)

# Read age data and map country names
age_df = pd.read_csv('./age.csv', index_col=0)
age_df = age_df.append({'country': 'Senegal', 'code': 'xxx', 'year': 2002, 'mean_age': 12}, ignore_index=True)
age_df = age_df.append({'country': 'Saudi Arabia', 'code': 'xxx', 'year': 2002, 'mean_age': 12}, ignore_index=True)
age_df.loc[:, 'country'] = age_df['country'].apply(lambda x: country_mapping[x] if x in country_mapping else x)


def get_score_cols(game, pred_scores):
    """Decide which columns to base calculations on."""
    if pred_scores == 'always':
        home_score_col, away_score_col = 'pred_home_score', 'pred_away_score'
    elif pred_scores == 'unplayed_only':
        if game['home_score'] is not None and game['away_score'] is not None:
            home_score_col, away_score_col = 'home_score', 'away_score'
        else:
            home_score_col, away_score_col = 'pred_home_score', 'pred_away_score'
    elif pred_scores == 'never':
        home_score_col, away_score_col = 'home_score', 'away_score'
    else:
        print("The parameter 'pred_scores' must be one of {'always', 'never', 'unplayed_only'}")
        raise
    return home_score_col, away_score_col


def get_all_games(team, df, pred):
    """Return a dictionary with all results for a team within the given dataframe."""
    tmp_results = {}
    for i, row in df.iterrows():
        if team in [row.home_team, row.away_team]:
            
            # Decide which columns to base calculations on
            home_score_col, away_score_col = get_score_cols(row, pred)
            
            # Add scores
            if row.home_team == team:
                tmp_results[row.date] = (row[home_score_col], row[away_score_col])
            else:
                tmp_results[row.date] = (row[away_score_col], row[home_score_col])

    return tmp_results


def create_results_per_team_df(df, pred='never'):
    """Create a result dataframe with teama as rows and rounds as columns."""
    team_df = pd.DataFrame()
    # Loop over all teams
    for team in sorted(df.home_team.append(df.away_team).unique()):
        team_df = team_df.append(pd.Series(get_all_games(team=team, df=df, pred=pred), name=team))
    # Reorder columns
    team_df = team_df[[col for col in sorted(team_df.columns)]]
    return team_df


def get_prev_results(row, team_df):
    """Return a pandas series with all results from games played earlier than the given row."""
    prev_results = team_df.loc[[row.home_team, row.away_team], team_df.columns < row.date]
    prev_results = prev_results.apply(lambda x: x.dropna().reset_index(drop=True), axis=1)
    prev_results = prev_results.apply(lambda x: [round(i, GOAL_DECIMALS) if isinstance(i, float) else i for j in x for i in j])
    prev_results = pd.Series({'round_{}'.format(i + 1): prev_results.get(i) for i in range(6)})
    return prev_results


def create_master_df(df, from_year=1998, to_year=2014, tournaments=['FIFA World Cup']):
    """Enrich dataframe with previous results and columns for predicted results."""
    
    # Disable SettingWithCopyWarning temporarily since we do nothing wrong
    pd.options.mode.chained_assignment = None
    
    master_df = pd.DataFrame()
    for year in range(from_year, to_year + 1, 4):

        # Subset to tournament and year
        world_cup_year = df[(df.tournament.apply(lambda x: x in tournaments)) & (df.date.apply(lambda x: x[:4]) == str(year))]
        world_cup_year['year'] = year

        # Add columns
        world_cup_year.loc[:, 'pred_home_score'] = world_cup_year.loc[:, 'pred_away_score'] = None
        for i in range(6):
            world_cup_year.loc[:, 'round_{}'.format(i + 1)] = None

        # Create results-per-team dataframe
        team_df = create_results_per_team_df(world_cup_year)

        # Get previous results for both teams per game
        prev_results = world_cup_year.apply(get_prev_results, axis=1, team_df=team_df)
        world_cup_year.loc[:, 'round_1':'round_6'] = prev_results

        # Append to world cup df
        master_df = master_df.append(world_cup_year)
    
    # Enable SettingWithCopyWarning again
    pd.options.mode.chained_assignment = 'warn'
    
    return master_df


def get_experience(row):
    """Return a list of experience features for the two teams in the row."""
    home_experience = experience_df.loc[(experience_df.cnty == row.home_team) & (experience_df.year == row.year)].values.squeeze()[3:]
    away_experience = experience_df.loc[(experience_df.cnty == row.away_team) & (experience_df.year == row.year)].values.squeeze()[3:]
    return list(np.concatenate((home_experience, away_experience)))


def get_age(row):
    """Return a list of mean age for the two teams in the row."""
    home_age = age_df.loc[(age_df.country == row.home_team) & (age_df.year == row.year), 'mean_age'].iloc[0]
    away_age = age_df.loc[(age_df.country == row.away_team) & (age_df.year == row.year), 'mean_age'].iloc[0]
    return [home_age, away_age]


def get_host_team(row):
    """Return a list indicating if any of the two teams is the host country."""
    return [float(row.home_team == row.country), float(row.away_team == row.country)]


def form_matrices(df):
    """Return matrices containing results, metadata, and correct results respectively."""
    # Results
    res = df.loc[:, df.columns.to_series().apply(lambda x: 'round' in x)]
    res = res.applymap(lambda entry: np.array(entry) if isinstance(entry, list) else np.array([99, 99, 99, 99]))
    res = np.array([[[value for value in prev] for prev in game[1]] for game in res.iterrows()])
    
    # Metadata
    meta = df.apply(lambda row: row['experience'] + row['age'] + row['host_team'], axis=1)
    meta = np.array([value for key, value in meta.items()])
    
    # Results
    try:
        y = df[['home_score', 'away_score']].values
    except KeyError:
        y = None
    
    return res, meta, y


def build_world_cup_predictor(x_res_shape, x_meta_shape, mask_value=99.0, r=0.1, n_lstm_cells=8, hidden_sizes=[8, 4]):
    """Define a model combining an LSTM with static input in a final NN."""
    # Clear tensorflow session
    backend.clear_session()
    
    # LSTM part
    result_input = Input(shape=(x_res_shape[1], x_res_shape[2]))
    masking = Masking(mask_value=mask_value)(result_input)
    lstm_out = LSTM(4, kernel_regularizer=regularizers.l2(r), bias_regularizer=regularizers.l2(r), activation='tanh')(masking)
    
    # Static part
    meta_input = Input(shape=(x_meta_shape[1],))
    
    # Concatenate
    hidden = [concatenate([lstm_out, meta_input])]
    for s in hidden_sizes:
        hidden.append(Dense(s, kernel_regularizer=regularizers.l2(r), bias_regularizer=regularizers.l2(r), activation='tanh')(hidden[-1]))
    output = Dense(2, kernel_regularizer=regularizers.l2(r), bias_regularizer=regularizers.l2(r), activation='relu')(hidden[-1])
    model = Model(inputs=[result_input, meta_input], outputs=output)
    
    return model


def train_world_cup_predictor(model, x_res, x_meta, y, validation_split=0.2, verbose=0):
    """Train the predictor model."""
    # Print timestamp
    timestamp = strftime('%Y%m%d-%H%M%S')
    print('Timestamp: {}'.format(timestamp))
    
    # Define loss, optimizer and metrics
    model.compile(loss='mse', optimizer='nadam', metrics=['mean_absolute_error'])

    # Define tensorboard callback
    tensorboard = TensorBoard(log_dir='./models/{}'.format(timestamp), histogram_freq=0, write_graph=True, write_images=False)
    
    # Train model
    try:
        model.fit(
            [x_res, x_meta],
            y,
            batch_size=int(x_res.shape[0] * (1 - validation_split)),  # BGD just because it's feasible
            epochs=20000,
            validation_split=validation_split,
            callbacks=[tensorboard],
            verbose=verbose
        )
    # Allow early stopping
    except KeyboardInterrupt:
        print('KeyboardInterrupt')

    return model, timestamp


def calculate_round(df):
    """Calculate round for each game in the dataframe."""
    game_round = df.apply(lambda row: df[((df.home_team == row.home_team) | (df.away_team == row.home_team)) & (df.date < row.date)].shape[0], axis=1)
    return game_round


def simulate_group_stage(df, model, pred='unplayed_only'):
    """Simulate all games in the group stage and return updated dataframe."""
    for i in [1, 2, 3]:
        # Form x matrices for round i
        res_tmp, meta_tmp, _ = form_matrices(df.loc[df.game_day == i])
        
        # Predict round i
        df.loc[df.game_day == i, ['pred_home_score', 'pred_away_score']] = model.predict([res_tmp, meta_tmp]).round(GOAL_DECIMALS)
    
        # Create results-per-team dataframe
        team_df = create_results_per_team_df(df, pred=pred)
    
        # Update previous results for all games since there are new results
        prev_results = df.apply(get_prev_results, axis=1, team_df=team_df)
        
        # Update dataframe
        df.loc[df.game_day > i, 'round_1':'round_6'] = prev_results
        
        # Replace NaN with None
        df = df.where(df.notnull(), None)
    
    return df


def _sum_up_group(group, win_factor=WIN_FACTOR, pred_scores='unplayed_only'):
    """Calculate points, scored and conceded goals, and goal difference within a group dataframe."""
    # Create empty group dictionary
    group_dict = defaultdict(lambda: defaultdict(int))
    
    # Loop over games within group
    for i, row in group.iterrows():
        
        # Decide which columns to base calculations on
        home_score_col, away_score_col = get_score_cols(row, pred_scores)
        tmp_win_factor = 1.0 if home_score_col == 'home_score' else win_factor
        
        # Update scored, conceded and goal difference
        group_dict[row.home_team]['scored'] += row[home_score_col]
        group_dict[row.home_team]['conceded'] += row[away_score_col]
        group_dict[row.home_team]['diff'] += row[home_score_col] - row[away_score_col]
        group_dict[row.away_team]['scored'] += row[away_score_col]
        group_dict[row.away_team]['conceded'] += row[home_score_col]
        group_dict[row.away_team]['diff'] += row[away_score_col] - row[home_score_col]
        # Update points
        if row[home_score_col] > (tmp_win_factor * row[away_score_col]):
            group_dict[row.home_team]['points'] += 3
        elif row[away_score_col] > (tmp_win_factor * row[home_score_col]):
            group_dict[row.away_team]['points'] += 3
        else:
            group_dict[row.home_team]['points'] += 1
            group_dict[row.away_team]['points'] += 1
    return group_dict


def sum_up_groups(df, pred_scores):
    """Loop over all groups and sum them up."""
    all_group_tables = {}
    for group, table in df.groupby('group').apply(_sum_up_group, pred_scores=pred_scores).items():
        all_group_tables[group] = pd.DataFrame(table).T.fillna(0)[['scored', 'conceded', 'diff', 'points']].sort_values(['points', 'diff', 'scored'], ascending=False)
    return all_group_tables


def simulate_round_of_16(df, model, group_tables, round_of_16, round_of_16_dates, pred_scores='unplayed_only'):
    """Simulate all matches in the round of 16 and return updated dataframe."""
    for match_nr, match in round_of_16.items():
        
        # Find the two teams at respective group positions
        team_1 = group_tables[match[0]].index[match[1]-1]
        team_2 = group_tables[match[2]].index[match[3]-1]
        
        # Start building new row for match
        new_row = pd.Series({
            'home_team': team_1,
            'away_team': team_2,
            'home_score': None,
            'away_score': None,
            'tournament': 'Fifa World Cup',
            'city': None,
            'country': df.country.unique()[0],
            'year': int(round_of_16_dates[match_nr - 1][:4]),
            'date': round_of_16_dates[match_nr - 1],
            'group': '1/8: {}'.format(match_nr)
        })
        new_row['experience'] = get_experience(new_row)
        new_row['age'] = get_age(new_row)
        new_row['host_team'] = get_host_team(new_row)
        team_df = create_results_per_team_df(df, pred=pred_scores)
        prev_results = get_prev_results(new_row, team_df)
        for i in range(6):
            new_row['round_{}'.format(i + 1)] = prev_results.get(i)
        
        # Form matrices, predict results, and add to new row
        tmp_res, tmp_meta, _ = form_matrices(new_row.to_frame().T)
        pred_res = model.predict([tmp_res, tmp_meta])
        new_row['pred_home_score'], new_row['pred_away_score'] = pred_res[0].round(GOAL_DECIMALS)
        
        # Add new row to master dataframe
        df = df.append(new_row, ignore_index=True)
        
        # Replace NaN with None
        df = df.where(df.notnull(), None)
    
    return df


def simulate_quarter_finals(df, model, quarter_finals, quarter_final_dates, pred_scores='unplayed_only'):
    """Simulate all quarter finals and return updated dataframe."""
    for match_letter, match in quarter_finals.items():

        # Find the two teams in the round of 16
        game_1 = df[df.group == '1/8: {}'.format(match[0])].iloc[0].to_dict()
        home_score_col, away_score_col = get_score_cols(game_1, pred_scores)
        team_1 = game_1['home_team'] if game_1[home_score_col] > game_1[away_score_col] else game_1['away_team']
        game_2 = df[df.group == '1/8: {}'.format(match[1])].iloc[0].to_dict()
        home_score_col, away_score_col = get_score_cols(game_2, pred_scores)
        team_2 = game_2['home_team'] if game_2[home_score_col] > game_2[away_score_col] else game_2['away_team']

        # Start building new row for match
        new_row = pd.Series({
            'home_team': team_1,
            'away_team': team_2,
            'home_score': None,
            'away_score': None,
            'tournament': 'Fifa World Cup',
            'city': None,
            'country': df.country.unique()[0],
            'year': int(quarter_final_dates[ord(match_letter) - ord('A')][:4]),
            'date': quarter_final_dates[ord(match_letter) - ord('A')],
            'group': '1/4: {}'.format(match_letter)
        })
        new_row['experience'] = get_experience(new_row)
        new_row['age'] = get_age(new_row)
        new_row['host_team'] = get_host_team(new_row)
        team_df = create_results_per_team_df(df, pred=pred_scores)
        prev_results = get_prev_results(new_row, team_df)
        for i in range(6):
            new_row['round_{}'.format(i + 1)] = prev_results.get(i)

        # Form matrices, predict results, and add to new row
        tmp_res, tmp_meta, _ = form_matrices(new_row.to_frame().T)
        pred_res = model.predict([tmp_res, tmp_meta])
        new_row['pred_home_score'], new_row['pred_away_score'] = pred_res[0].round(GOAL_DECIMALS)

        # Add new row to master dataframe
        df = df.append(new_row, ignore_index=True)
        
        # Replace NaN with None
        df = df.where(df.notnull(), None)

    return df


def simulate_semi_finals(df, model, semi_finals, semi_final_dates, pred_scores='unplayed_only'):
    """Simulate both semi finals and return updated dataframe."""
    for match_nr, match in semi_finals.items():
        
        # Find the two teams in the quarter finals
        game_1 = df[df.group == '1/4: {}'.format(match[0])].iloc[0].to_dict()
        home_score_col, away_score_col = get_score_cols(game_1, pred_scores)
        team_1 = game_1['home_team'] if game_1[home_score_col] > game_1[away_score_col] else game_1['away_team']
        game_2 = df[df.group == '1/4: {}'.format(match[1])].iloc[0].to_dict()
        home_score_col, away_score_col = get_score_cols(game_2, pred_scores)
        team_2 = game_2['home_team'] if game_2[home_score_col] > game_2[away_score_col] else game_2['away_team']

        # Start building new row for match
        new_row = pd.Series({
            'home_team': team_1,
            'away_team': team_2,
            'home_score': None,
            'away_score': None,
            'tournament': 'Fifa World Cup',
            'city': None,
            'country': df.country.unique()[0],
            'year': int(semi_final_dates[match_nr - 1][:4]),
            'date': semi_final_dates[match_nr - 1],
            'group': '1/2: {}'.format(match_nr)
        })
        new_row['experience'] = get_experience(new_row)
        new_row['age'] = get_age(new_row)
        new_row['host_team'] = get_host_team(new_row)
        team_df = create_results_per_team_df(df, pred=pred_scores)
        prev_results = get_prev_results(new_row, team_df)
        for i in range(6):
            new_row['round_{}'.format(i + 1)] = prev_results.get(i)

        # Form matrices, predict results, and add to new row
        tmp_res, tmp_meta, _ = form_matrices(new_row.to_frame().T)
        pred_res = model.predict([tmp_res, tmp_meta])
        new_row['pred_home_score'], new_row['pred_away_score'] = pred_res[0].round(GOAL_DECIMALS)

        # Add new row to master dataframe
        df = df.append(new_row, ignore_index=True)
        
        # Replace NaN with None
        df = df.where(df.notnull(), None)

    return df


def simulate_finals(df, model, finals, final_dates, pred_scores='unplayed_only'):
    """Simulate gold and broze match and return updated dataframe."""
    for match_name, match in finals.items():
        
        # Find the two teams in the semi finals
        game_1 = df[df.group == '1/2: {}'.format(match[0])].iloc[0].to_dict()
        game_2 = df[df.group == '1/2: {}'.format(match[1])].iloc[0].to_dict()
        if match_name == 'gold':
            home_score_col, away_score_col = get_score_cols(game_1, pred_scores)
            team_1 = game_1['home_team'] if game_1[home_score_col] > game_1[away_score_col] else game_1['away_team']
            home_score_col, away_score_col = get_score_cols(game_2, pred_scores)
            team_2 = game_2['home_team'] if game_2[home_score_col] > game_2[away_score_col] else game_2['away_team']
        else:
            home_score_col, away_score_col = get_score_cols(game_1, pred_scores)
            team_1 = game_1['home_team'] if game_1[home_score_col] < game_1[away_score_col] else game_1['away_team']
            # team_1 = game_1['home_team'] if game_1['pred_home_score'] < game_1['pred_away_score'] else game_1['away_team']
            home_score_col, away_score_col = get_score_cols(game_2, pred_scores)
            team_2 = game_2['home_team'] if game_2[home_score_col] < game_2[away_score_col] else game_2['away_team']

        # Start building new row for match
        new_row = pd.Series({
            'home_team': team_1,
            'away_team': team_2,
            'home_score': None,
            'away_score': None,
            'tournament': 'Fifa World Cup',
            'city': None,
            'country': df.country.unique()[0],
            'year': int(final_dates[match_name == 'gold'][:4]),
            'date': final_dates[match_name == 'gold'],
            'group': match_name
        })
        new_row['experience'] = get_experience(new_row)
        new_row['age'] = get_age(new_row)
        new_row['host_team'] = get_host_team(new_row)
        team_df = create_results_per_team_df(df, pred=pred_scores)
        prev_results = get_prev_results(new_row, team_df)
        for i in range(6):
            new_row['round_{}'.format(i + 1)] = prev_results.get(i)

        # Form matrices, predict results, and add to new row
        tmp_res, tmp_meta, _ = form_matrices(new_row.to_frame().T)
        pred_res = model.predict([tmp_res, tmp_meta])
        new_row['pred_home_score'], new_row['pred_away_score'] = pred_res[0].round(GOAL_DECIMALS)
        
        # Add new row to master dataframe
        df = df.append(new_row, ignore_index=True)
        
        # Replace NaN with None
        df = df.where(df.notnull(), None)

    return df

import time
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
from db import get_heros
import matplotlib
matplotlib.use('Agg')
import matplotlib.cm as cm
import tensorflow as tf
import pickle
import numpy as np
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf
import matplotlib.pyplot as plt
import io
import base64
from flask_cors import CORS
app = Flask(__name__, static_folder='../frontend/dist', static_url_path='/')
CORS(app, resources={r"/*": {"origins": "*"}}, supports_credentials=True)


class DraftPredictor:
    def __init__(self, models, hero_to_idx, idx_to_hero, n_features=254, max_time_steps=10):
        self.models = models
        self.hero_to_idx = hero_to_idx
        self.idx_to_hero = idx_to_hero
        self.n_features = n_features  # Should be len(hero_to_idx) * 2
        self.max_time_steps = max_time_steps  # From data preparation
    def encode_draft_state(self, current_picks):
        """Encode the current draft state into a one-hot encoded vector."""
        num_heroes = len(self.hero_to_idx)
        state = np.zeros(self.n_features)
        for team, hero in current_picks:
            if hero in self.hero_to_idx:
                idx = self.hero_to_idx[hero]
                if team == 'our':
                    state[idx] = 1  # Our picks
                else:
                    state[num_heroes + idx] = 1  # Enemy picks
            else:
                print(f"Warning: Hero {hero} not found in hero_to_idx")
        return state
    def prepare_sequence(self, our_picks, enemy_picks):
        """Prepare the draft sequence for model prediction."""

        # Build the sequence of states up to the current picks
        num_heroes = len(self.hero_to_idx)
        sequence = []
        current_picks = []

        # Define the pick order as per the data preparation
        pick_order = [
            ('our', 1),    # Blue Pick 1
            ('enemy', 2),
            ('our', 2),
            ('enemy', 2),
            ('our', 2),
            ('enemy', 1)
        ]

        our_picks_iter = iter(our_picks)
        enemy_picks_iter = iter(enemy_picks)

        # Build the picks_sequence based on the picks we have so far
        for team, num_picks in pick_order:
            for _ in range(num_picks):
                if team == 'our':
                    try:
                        hero = next(our_picks_iter)
                    except StopIteration:
                        hero = None
                else:
                    try:
                        hero = next(enemy_picks_iter)
                    except StopIteration:
                        hero = None

                if hero is not None:
                    current_picks.append((team, hero))
                else:
                    # No more picks from this team
                    break

                # Encode current state
                state = self.encode_draft_state(current_picks[:-1])  # Exclude current pick
                sequence.append(state)

        # Pad the sequence to max_time_steps
        padded_sequence = np.zeros((self.max_time_steps, self.n_features))
        for i, state in enumerate(sequence):
            padded_sequence[i] = state

        return np.expand_dims(padded_sequence, axis=0)  # Shape (1, max_time_steps, n_features)
    def decode_predictions(self, predictions, current_picks, top_n=10, banned_heroes=[]):
        """Convert model predictions to hero recommendations."""
        num_heroes = len(self.hero_to_idx)
        picks_so_far = len(current_picks)
        if picks_so_far == 0:
            timestep = 0
        else:
            timestep = picks_so_far - 1
        next_pick_probs = predictions[0, timestep]  # Get prediction at current timestep
        picked_heroes = set([hero for team, hero in current_picks] + banned_heroes)
        hero_probs = []
        for idx, prob in enumerate(next_pick_probs):
            hero = self.idx_to_hero[idx]
            if hero not in picked_heroes:
                hero_probs.append((hero, prob))

        # Sort and get top N
        sorted_heroes = sorted(hero_probs, key=lambda x: x[1], reverse=True)

        return sorted_heroes[:top_n]
    def get_recommendations(self, our_picks, enemy_picks, top_n=10, banned_heroes=[]):
        """Generate a unified bar chart combining recommendations from all models."""
        current_picks = []
        for hero in our_picks:
            current_picks.append(('our', hero))
        for hero in enemy_picks:
            current_picks.append(('enemy', hero))

        draft_sequence = self.prepare_sequence(our_picks, enemy_picks)

        # Gather all model predictions
        hero_scores = {}
        n_models = len(self.models)
        # Use a color palette that works well for stacked bars
        colors = plt.cm.viridis(np.linspace(0, 1.1, n_models))

        for model_idx, (model_name, model_data) in enumerate(self.models.items()):
            predictions = model_data["model"].predict(draft_sequence, verbose=0)
            top_heroes = self.decode_predictions(predictions, current_picks, top_n=None, banned_heroes=banned_heroes)

            for hero, confidence in top_heroes:
                if hero not in hero_scores:
                    hero_scores[hero] = np.zeros(n_models)
                hero_scores[hero][model_idx] = confidence

        # Prepare data for plotting
        hero_scores_sorted = sorted(hero_scores.items(), key=lambda x: np.sum(x[1]), reverse=True)[:top_n]
        heroes, scores = zip(*hero_scores_sorted)
        scores = np.array(scores)

        # Calculate figure size based on content
        base_height = max(8, min(12, 0.5 * top_n))  # Dynamic height based on number of heroes
        # Dynamically set figure size
        base_width = 15
        legend_width = 3
        total_width = base_width + legend_width
        fig = plt.figure(figsize=(total_width, base_height))

        # Create main axis for bars
        ax = plt.gca()
        
        # Plot stacked bars
        cumulative_scores = np.zeros(len(heroes))
        bars = []
        model_names = list(self.models.keys())
        
        for model_idx in range(n_models):
            model_contributions = scores[:, model_idx]
            bar = ax.barh(
                heroes,
                model_contributions,
                left=cumulative_scores,
                color=colors[model_idx],
                alpha=0.8,
                label=model_names[model_idx],
                edgecolor='white',
                linewidth=0.5
            )
            bars.append(bar)
            cumulative_scores += model_contributions

            # Add value labels for significant contributions
            for idx, (width, y) in enumerate(zip(model_contributions, heroes)):
                if width > 0.05:  # Only show if contribution is significant
                    x = cumulative_scores[idx] - width/2
                    ax.text(x, idx, f'{width:.2f}', 
                        ha='center', va='center',
                        color='white', fontsize=8, fontweight='bold')

        # Style improvements
        ax.set_title("Hero Recommendations Across Models", 
                    fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel("Cumulative Confidence Score", fontsize=12, labelpad=10)
        ax.set_ylabel("Hero Recommendations", fontsize=12, labelpad=10)

        # Customize grid
        ax.grid(axis='x', linestyle='--', alpha=0.2, zorder=0)
        ax.set_axisbelow(True)

        # Remove unnecessary spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # Invert y-axis and adjust layout
        ax.invert_yaxis()

        # Create primary legend for models
        legend1 = ax.legend(
            title="Models",
            loc='center left',  # Place it inside the plot on the right
            bbox_to_anchor=(1.05, 0.5),  # Adjust the position inside
            fontsize=10,
            title_fontsize=12
        )
        ax.add_artist(legend1)
        # Create draft information legend
        """
        draft_labels = [
            f"Our Team: {', '.join(our_picks)}",
            f"Enemy Team: {', '.join(enemy_picks)}",
            f"Banned: {', '.join(banned_heroes)}"
        ]
        
        # Create proxy artists for the draft legend
        draft_patches = [plt.Rectangle((0, 0), 1, 1, fc='none', fill=False, edgecolor='none', linewidth=0) 
                        for _ in draft_labels]
        
        # Add second legend below the first one
        
        draft_legend = ax.legend(
            draft_patches, draft_labels,
            loc='lower left',  # Place below the first legend
            bbox_to_anchor=(1.05, 0.5),  # Adjust this anchor point
            fontsize=10,
            title_fontsize=12
        )
        
        # Add both legends to the plot
        ax.add_artist(legend1)
        ax.add_artist(draft_legend)
        """
        # Adjust layout to prevent text cutoff
        plt.subplots_adjust(left=.1, right=0.75, bottom=0.2)  # Increase right margin and bottom margin

        
        # Add total confidence scores at the end of each bar
        for idx, total in enumerate(cumulative_scores):
            ax.text(total + 0.01, idx, f'Total: {total:.2f}',
                    va='center', ha='left', fontsize=9,
                    color='dimgray', fontweight='bold')

        # Save the figure to a BytesIO buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches=None)
        buf.seek(0)

        # Encode the buffer as Base64
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')
        buf.close()
        plt.close(fig)  # Close the figure to free resources
        return img_base64
# Load hero_to_idx and idx_to_hero mappings
with open('Trained_Models\\hero_to_idx.pkl', 'rb') as f:
    hero_to_idx = pickle.load(f)

with open('Trained_Models\\idx_to_hero.pkl', 'rb') as f:
    idx_to_hero = pickle.load(f)
# Number of features
n_heroes = len(hero_to_idx)
n_features = n_heroes * 2  # Our picks + Enemy picks

# Max time steps as determined during data preparation
max_time_steps = 10  # As per the draft length in the data preparation

# Number of targets
n_targets = n_heroes  # We predict over all heroes

# Load the trained models
models = {}

model_files = {
    "Model 1: Deep_LSTM": 'model_1_deep_lstm_best.keras',
    "Model 2: Deep_LSTM": 'model_2_deep_lstm_best.keras',
    "Model 3: Deep_LSTM": 'model_3_deep_lstm_best.keras',
    "Model 4: Deep_LSTM": 'model_4_deep_lstm_best.keras',
    "Model 5: Enhanced_Conv1D_LSTM": 'model_5_enhanced_conv1d_lstm_best.keras',
    "Model 6: Enhanced_Conv1D_LSTM": 'model_6_enhanced_conv1d_lstm_best.keras',
    "Model 7: Parallel_Conv1D_GRU": 'model_7_parallel_conv1d_gru_best.keras',
    "Model 8: Parallel_Conv1D_GRU": 'model_8_parallel_conv1d_gru_best.keras',
    "Model 9: Deep_Conv1D": 'model_9_deep_conv1d_best.keras',
    "Model 10: Deep_Conv1D": 'model_10_deep_conv1d_best.keras',
    "Model 11: rb_Deep_LSTM": 'special_11_deep_lstm_best.keras',
    "Model 12: rb_Enhanced_Conv1D_LSTM": 'special_12_enhanced_conv1d_lstm_best.keras',
    "Model 13: rb_Parallel_Conv1D_GRU": 'special_13_parallel_conv1d_gru_best.keras',
    "Model 14: rb_Deep_Conv1D": 'special_14_deep_conv1d_best.keras',
    "Model 15: BIG_LSTM": 'BIG_LSTM_model_15_deep_lstm_best.keras',
    "Model 16: BIG_ENCHANCED": 'BIG_ENCHANCED_model_16_enhanced_conv1d_lstm_best.keras',
    'Model 17: BIG_GRU': 'BIG_GRU_model_17_parallel_conv1d_gru_best.keras',
    "Model 18: BIG_CONV1D": 'BIG_CONV1D_model_18_deep_conv1d_best.keras'

}

for model_name, model_file in model_files.items():
    model = tf.keras.models.load_model('Trained_Models\\' + model_file)
    models[model_name] = {
        "model": model
    }

predictor = DraftPredictor(
    models=models,
    hero_to_idx=hero_to_idx,
    idx_to_hero=idx_to_hero,
)

def filter_picks(picks):
    """Utility function to filter empty picks."""
    return [pick for pick in picks if pick]

@app.route('/api/heroes', methods=['GET'])
def get_heroes():
    hero_list = get_heros()
    return jsonify({"message": hero_list})

@app.route('/api/test', methods=['GET'])
def test_route():
    return jsonify({"message": "Backend is working!"})

@app.route('/api/predictions', methods=['POST'])
def pred(our_picks=[], enemy_picks=[], banned_heroes=[], top_n=10):
    # Our picks and enemy picks so far
    #our_picks = ["Suyou", "Yu Zhong", "Valentina", "Claude", ""]  # Our team's picks
    #enemy_picks = ["Bruno", "Gatotkaca", "Joy", "Khaleed", ""]  # Enemy team's picks
    # Banned heroes
    #banned_heroes = ["default", "Ling", "Tigreal", "Fanny", "Chip", "Joy", "Arlott", "Granger", "Suyou", "", "", ""]
    # Filter out empty banned heroes
    # Default empty lists for optional arguments
    data = request.json
    our_picks = data.get('our_picks', [])
    enemy_picks = data.get('enemy_picks', [])
    banned_heroes = data.get('banned_heroes', [])
    top_n = data.get('top_n', 10)
    
    our_picks = filter_picks(our_picks or [])
    enemy_picks = filter_picks(enemy_picks or [])
    banned_heroes = filter_picks(banned_heroes or [])

    print("Our Picks:", our_picks)
    print("Enemy Picks:", enemy_picks)
    print("Banned Heroes:", banned_heroes)

    banned_heroes.append('default')
    
    # Generate recommendations
    recommendations = {
        "ourPick": predictor.get_recommendations(
            our_picks=our_picks, enemy_picks=enemy_picks, top_n=top_n, banned_heroes=banned_heroes
        ),
        "enemyPick": predictor.get_recommendations(
            our_picks=enemy_picks, enemy_picks=our_picks, top_n=top_n, banned_heroes=banned_heroes
        ),
    }
    return jsonify(recommendations)

@app.route('/api/feedback', methods=['POST'])
def provide_feedback():
    # Get data from request
    data = request.json
    our_picks = data.get('our_picks', [])
    enemy_picks = data.get('enemy_picks', [])
    banned_heroes = data.get('banned_heroes', [])
    target_pick = data.get('target_pick', None)
    
    # Preprocess picks
    our_picks = filter_picks(our_picks or [])
    enemy_picks = filter_picks(enemy_picks or [])
    banned_heroes = filter_picks(banned_heroes or [])
    banned_heroes.append('default')
    
    # Ensure target_pick is provided and valid
    if not target_pick:
        return jsonify({"error": "target_pick is required"}), 400

    # If target_pick is a list, extract the first element
    if isinstance(target_pick, list):
        if len(target_pick) == 0:
            return jsonify({"error": "target_pick is empty"}), 400
        target_pick = target_pick[0]

    if target_pick not in predictor.hero_to_idx:
        return jsonify({"error": f"Invalid target_pick: {target_pick}"}), 400

    print("Our Picks:", our_picks)
    print("Enemy Picks:", enemy_picks)
    print("Banned Heroes:", banned_heroes)
    print("Target Pick (Human Feedback):", target_pick)
    print("Type of target_pick:", type(target_pick))

    save = {}
    save = {
    'our_picks': our_picks,
    'enemy_picks': enemy_picks,
    'banned': banned_heroes,
    'target': target_pick
    }
    save_file = 'Trained_Models\\to_train_dict.pkl'

    if os.path.exists(save_file):
        # Load existing data and append
        with open(save_file, 'rb') as file:
            existing_data = pickle.load(file)
        
        if isinstance(existing_data, list):
            existing_data.append(save)
        else:
            # If existing data is not a list, convert it to a list
            existing_data = [existing_data, save]
    else:
        # If the file does not exist, create a new list
        existing_data = [save]

    # Save the updated data back to the pickle file
    with open(save_file, 'wb') as file:
        print("To train:\n"+save_file)
        pickle.dump(existing_data, file)
    time.sleep(.5) #just for show
    return jsonify({"message": "Feedback received! Models will be updated every Sunday night @ 11:59 PM (EST)"}), 200
"""
    # Prepare the input sequence
    draft_sequence = predictor.prepare_sequence(our_picks, enemy_picks)
    
    # Prepare the target
    num_heroes = len(predictor.hero_to_idx)
    # Create target with shape (1, max_time_steps, num_heroes)
    target = np.zeros((1, predictor.max_time_steps, num_heroes))
    
    # Determine the current timestep
    picks_so_far = len(our_picks) + len(enemy_picks)
    if picks_so_far == 0:
        timestep = 0
    else:
        timestep = picks_so_far - 1

    # Set the target pick at the current timestep
    target_idx = predictor.hero_to_idx[target_pick]
    target[0, timestep, target_idx] = 1
    
    #save new models + train them
    for model_name, model_file in model_files.items():

        print(f"Updating model: {model_name}")
        model = models[model_name].get("model")
        model.fit(draft_sequence, target, epochs=1, verbose=1)
        model.save('Trained_Models\\' + model_file)

    return jsonify({"message": "Feedback received and models updated"}), 200
"""
##TODO: add scraping capability in mlbb_work folder
@app.route("/", defaults={'path': ''})
@app.route("/<path:path>")
def serve(path):
    if path != "" and os.path.exists(os.path.join(app.static_folder, path)):
        return send_from_directory(app.static_folder, path)
    else:
        return send_from_directory(app.static_folder, 'index.html')



if __name__ == '__main__':
    app.run(use_reloader=True, threaded=True)
"""if __name__ == '__main__':
    app.run(debug=True, port=5000)"""
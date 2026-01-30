import sys
import os

# Add project root to path
base_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../'))
sys.path.append(base_dir)

from playground.analysis_utils import plot_states_from_csv, plot_model_input_from_csv

# Get the CSV file path relative to this script
#csv_filepath = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'pendulum_states.csv')
csv_filepath = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'pendulum_model_inputs.csv')

#plot_states_from_csv(csv_filepath, 250)
plot_model_input_from_csv(csv_filepath, "states", 500)

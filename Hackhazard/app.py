from flask import Flask, jsonify, request, render_template
import pandas as pd
from dotenv import load_dotenv
import os

load_dotenv()  # This line loads the variables from .env file

import groq  # Groq API client

# Initialize Flask app
app = Flask(__name__, template_folder='templates')

groq_client = groq.Client(api_key=os.getenv('GROQ_API_KEY'))


def load_dataset(filename):
    """Helper function to load datasets with error handling."""
    if os.path.exists(filename):
        try:
            return pd.read_csv(filename, encoding='ISO-8859-1')
        except Exception as e:
            print(f"Error loading {filename}: {e}")
            return pd.DataFrame()
    else:
        print(f"File not found: {filename}")
        return pd.DataFrame()
    


laptop_data = load_dataset('laptops_data.csv')

# Helper function for filtering
def filter_data(data, filters):
    """Filter data based on the given filters."""
    filtered = data.copy()
    for key, value in filters.items():
        if value:
            if key == "Price Sort":
                filtered = filtered.sort_values("Price", ascending=(value == "asc"))
            elif key in data.columns:
                filtered = filtered[filtered[key].astype(str).str.contains(value, case=False, na=False)]
    return filtered

# Endpoint to provide category options for laptops
@app.route('/api/filters/laptops', methods=['GET'])
def laptop_filters():
    categories = ["Ultrabook", "Gaming", "Notebook", "2 in 1 Convertible"]
    return jsonify({"categories": categories})



@app.route('/api/laptops', methods=['GET'])
def laptops():
    if laptop_data.empty:
        return jsonify({"error": "Laptop data not available"}), 404

    # Extract filters from request
    category = request.args.get('category')
    filters = {}
    if category:
        filters['TypeName'] = category

    # Filter data
    filtered_data = filter_data(laptop_data, filters) if filters else laptop_data

    # Paginate the filtered data
    page = int(request.args.get('page', 1))
    per_page = 10
    start = (page - 1) * per_page
    end = start + per_page
    paginated_data = filtered_data.iloc[start:end]
    total_pages = (len(filtered_data) + per_page - 1) // per_page

    # Send the paginated, filtered data
    return jsonify({
        "laptops": paginated_data.to_dict(orient='records'),
        "total_pages": total_pages
    })




@app.route('/api/ai', methods=['POST'])
def ai_assistant():
    data = request.get_json()
    prompt = data.get("prompt")

    if not prompt:
        return jsonify({"error": "No prompt provided"}), 400

    try:
        response = groq_client.chat.completions.create(
            model="gemma2-9b-it",  # Ensure the model ID is correct
            messages=[
                {"role": "system", "content": "You are an AI expert helping with technology-related queries."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7
        )
        ai_response = response.choices[0].message.content
        return jsonify({"response": ai_response})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Route to render the homepage
@app.route('/')
def home():
    return render_template('index.html')

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)

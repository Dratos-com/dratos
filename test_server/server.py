from flask import Flask, request, jsonify
import requests
import json
import os

app = Flask(__name__)

# Load test data
current_dir = os.path.dirname(os.path.abspath(__file__))
test_data_path = os.path.join(current_dir, 'test_data.json')
with open(test_data_path, 'r') as f:
    test_data = json.load(f)

@app.route('/', methods=['POST'])
@app.route('/<path:any_path>', methods=['POST']) 
def api(any_path=None):
    request_data = request.json
    request_data["path"] = any_path
    request_key = json.dumps(request_data, sort_keys=True)
    if request_key in test_data:
        return jsonify(test_data[request_key])
    else:
        try:
            print("\033[94mMISSING KEY IN TEST_DATA.JSON, FORWARDING TO ACTUAL API\033[0m")
            # Get the target base URL from the environment variable
            engine = os.getenv("ENGINE")
            base_url = os.getenv(f"{engine}_BASE_URL")
            api_key = os.getenv(f"{engine}_API_KEY")

            if base_url:
                # Construct the full target URL
                full_target_url = f"{base_url.rstrip('/')}/{any_path}" if any_path else base_url

                # Prepare the request data (excluding 'path' key)
                payload = {k: v for k, v in request_data.items() if k != 'path'}

                # Get the API key from the environment variable
                if not api_key:
                    return jsonify({"error": f"{engine}_API_KEY is not set"}), 500

                # Prepare headers with API key
                headers = {
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json"
                }

                try:
                    # Send the request to the target URL with headers
                    response = requests.post(full_target_url, json=payload, headers=headers)
                    response_data = response.json()

                    # Store the response in test_data
                    test_data[request_key] = response_data

                    # Save updated test_data to file
                    with open(test_data_path, 'w') as f:
                        json.dump(test_data, f, indent=2)

                    return jsonify(response_data), response.status_code
                except requests.RequestException as e:
                    return jsonify({"error": f"Error forwarding request: {str(e)}"}), 500
            else:
                return jsonify({"error": "TARGET_BASE_URL is not set"}), 500
        except Exception as e:
            return jsonify({"error": f"No matching data found in test_data.json. Tried to forward request to actual API but: {e}"}), 404


if __name__ == '__main__':
    app.run(debug=True, port=4000)

import requests

token = 'UMDY2QH3zO7Rj803vKRu7exTsSnhftICxOXy4QO5'
deployment_url = "https://api.acc.deeploy.ml/v2/workspaces/bb97342f-b6de-4eee-ab68-0e8e72982587/deployments/ae6026c4-055d-4f0f-815d-505888697cbf/predict"

# Example: a batch prediction with two input tensors
model_input = {
  "instances":
      [[-0.6409383911408225, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.5921480257661339, 1.0]]
      }

headers = {
  'Authorization': 'Bearer %s' % token,
}
response = requests.post(deployment_url, headers=headers, json=model_input)
model_output = response.json()
print(model_output)
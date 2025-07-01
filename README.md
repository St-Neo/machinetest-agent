# machinetest-agent

Steps to run locally
Copy .env.local to .env and add the required environment variables. (not needed for main.py but will be needed for graph.py)

Install dependencies (using pip):

```pip install -r requirements.txt```
mighty be needed (not sure)
```pip install langgraph langchain_core  ```

uncoomment the run_scenario call to run the other mocks. parallel execution of all the topics is not done.
everything is mocked and hardcoded.
the console project can be run with:
```python main.py```

# DOPL
Direct Online Preference Learning

This is the official Codebase for the paper: Direct Online Preference Learning for Restless Bandits with Preference Feedback.

To run the Linear Solver Gurobi License is needed. Please refer to the link: [a link](https://www.gurobi.com/academia/academic-program-and-licenses/)

To run the code for a specific environment.
```python run.py --config-name=<env_name>``` 

For Example: If we want to run DOPL algorithm in the cpap environment:
```python run.py --config-name=cpap``` 

For CPAP environment: ```python create_cpap.py```
This creates the state dependent reward and transition probabilities for each arm.

To create an environment 

output will be stored in a timestamped directory in ```outputs``` folder

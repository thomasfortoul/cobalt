# hello



A subset of the codebase for the 5-part server based trading architecture which builds ML Models with 'trainer.py', downloads models to a joblib object, then produces predictions with 'prediction.py', and executes trades with the Interactive Broker API with 'trading.py'.
Data manipulation, collection is done through SQL queries on the data collector server. Auditing and surveillance is done by an additional node as well.

Model retraining was automatically setup monthly.
// config/mongodb/init.js — Initialisation MongoDB (exécuté une seule fois au premier démarrage)
db = db.getSiblingDB('disinformation_db');

db.createCollection('articles');
db.articles.createIndex({ id: 1 }, { unique: true });
db.articles.createIndex({ processed_at: -1 });
db.articles.createIndex({ is_fake: 1 });
db.articles.createIndex({ source: 1 });
db.articles.createIndex({ title: 'text', body: 'text' });

db.createCollection('drift_events');
db.drift_events.createIndex({ detected_at: -1 });
db.drift_events.createIndex({ detector: 1 });

db.createCollection('online_learning_log');
db.online_learning_log.createIndex({ timestamp: -1 });

print('disinformation_db initialisée : collections articles, drift_events, online_learning_log');

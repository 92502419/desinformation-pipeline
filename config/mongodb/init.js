// config/mongodb/init.js
// Exécuté automatiquement au démarrage du container MongoDB


db = db.getSiblingDB('disinformation_db');


// ── COLLECTION ARTICLES ──────────────────────────────────
db.createCollection('articles', {
  validator: {
    $jsonSchema: {
      bsonType: 'object',
      required: ['id', 'title', 'is_fake', 'processed_at'],
      properties: {
        id:          { bsonType: 'string' },
        title:       { bsonType: 'string' },
        body:        { bsonType: 'string' },
        url:         { bsonType: 'string' },
        source:      { bsonType: 'string' },
        is_fake:     { bsonType: 'int',    minimum: 0, maximum: 1 },
        confidence:  { bsonType: 'double', minimum: 0, maximum: 1 },
        p_fake:      { bsonType: 'double', minimum: 0, maximum: 1 },
        drift_score: { bsonType: 'double' },
        processed_at:{ bsonType: 'string' },
      }
    }
  },
  validationAction: 'warn'   // Avertir sans bloquer
});


// ── INDEX POUR LES REQUÊTES FRÉQUENTES ───────────────────
db.articles.createIndex({ 'processed_at': -1 });
db.articles.createIndex({ 'is_fake': 1, 'processed_at': -1 });
db.articles.createIndex({ 'source': 1 });
db.articles.createIndex({ 'drift_score': -1 });
db.articles.createIndex({ 'id': 1 }, { unique: true });


// ── COLLECTION DRIFT EVENTS ──────────────────────────────
db.createCollection('drift_events');
db.drift_events.createIndex({ 'timestamp': -1 });


print('MongoDB initialisé avec succès !');
print('Collections : articles, drift_events');
print('Index créés sur : processed_at, is_fake, source, drift_score, id');


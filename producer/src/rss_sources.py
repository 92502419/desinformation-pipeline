# producer/src/rss_sources.py


RSS_SOURCES = {
    # ── Sources fiables (fact-checkées, agences de presse reconnues) ──
    'reliable': [
        {'name': 'AFP',           'url': 'https://www.afp.com/fr/actualites/feeds/feed-afp-com.xml', 'lang': 'fr', 'category': 'reliable'},
        {'name': 'Reuters',       'url': 'https://feeds.reuters.com/reuters/topNews',                'lang': 'en', 'category': 'reliable'},
        {'name': 'BBC News',      'url': 'https://feeds.bbci.co.uk/news/world/rss.xml',              'lang': 'en', 'category': 'reliable'},
        {'name': 'Al Jazeera',    'url': 'https://www.aljazeera.com/xml/rss/all.xml',               'lang': 'en', 'category': 'reliable'},
        {'name': 'Le Monde',      'url': 'https://www.lemonde.fr/rss/une.xml',                      'lang': 'fr', 'category': 'reliable'},
        {'name': 'RFI',           'url': 'https://www.rfi.fr/fr/rss',                               'lang': 'fr', 'category': 'reliable'},
        {'name': 'AP News',       'url': 'https://rsshub.app/apnews/topics/apf-topnews',            'lang': 'en', 'category': 'reliable'},
        {'name': 'Deutsche Welle','url': 'https://rss.dw.com/xml/rss-en-world',                    'lang': 'en', 'category': 'reliable'},
        {'name': 'France24',      'url': 'https://www.france24.com/fr/rss',                        'lang': 'fr', 'category': 'reliable'},
        {'name': 'VOA News',      'url': 'https://www.voanews.com/rss/',                           'lang': 'en', 'category': 'reliable'},
        {'name': 'Jeune Afrique', 'url': 'https://www.jeuneafrique.com/feed/',                     'lang': 'fr', 'category': 'reliable'},
        {'name': 'Africa News',   'url': 'https://www.africanews.com/feed/rss',                    'lang': 'en', 'category': 'reliable'},
    ],
    # ── Sources à surveiller (potentiellement douteuses) ─────────────
    # Référence : liste NewsGuard / IFCN — à compléter via fichier externe
    'suspicious': []
}


# Toutes les sources combinées pour le scraping
ALL_SOURCES = RSS_SOURCES['reliable'] + RSS_SOURCES['suspicious']


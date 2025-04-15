### Sujet de Stage : Benchmark pour évaluer des architectures d'IA proto-LLM sur tâches séquentielles

#### Contexte et Problématique

La capacité à maintenir et manipuler des informations au fil du temps est un aspect essentiel de l'intelligence artificielle. Les modèles séquentiels, tels que les Réseaux Neuraux Récurrents (RNN) et les Transformers (architecture à la base des LLM), sont couramment utilisés pour des tâches nécessitant une mémoire de travail, comme le traitement du langage naturel et la prédiction de séries temporelles. Cependant, évaluer la capacité de ces modèles à maintenir et manipuler des informations sur de longues périodes reste un défi majeur : il n'existe pas aujourd'hui de tâches simples, standardisées et scalables permettant d'évaluer une architecture sur un ensemble de critères tout au long de son entrainement. Ainsi, lorsqu'un chercheur propose une nouvelle architecture, il ne peut l'évaluer que sur 2-3 tâches simples avant de devoir passer à du texte. Le gap entre les deux est grand et demande d'avoir accès très rapidement à une grosse puissance de calcul. Par ailleurs, ces tâches n'étant que trop peu souvent standardisée, la comparaison avec les modèles existants devient difficile.

Pour répondre à ces limitations, nous proposons un nouveau benchmark appelé STREAM (Sequential Tasks Review to Evaluate Artificial Memory). Ce benchmark vise à offrir une suite diversifiée de tâches scalable qui testent différentes facettes de la mémoire de travail artificielle (mémoire simple, traitement de signal, dépendances à long terme et manipulation d'informations). Ainsi, avec STREAM, les chercheurs pourront plus facilement évaluer, comparer et scaler leur architecture.

#### Objectifs du Stage

Le stagiaire aura pour mission principale de contribuer au développement et à l'expérimentation du benchmark STREAM. Les objectifs spécifiques incluent :

1. **Développement des Tâches du Benchmark** : Implémenter de nouvelles tâches et ajuster les 12 tâches déjà définies dans le benchmark STREAM, couvrant des domaines tels que la mémoire simple, le traitement de signaux, les dépendances à long terme et la manipulation d'informations.
2. **Évaluation des Modèles Existants** : Entraîner et tester divers modèles State-Of-The-Art (LSTM, Transformers, Transformer-Decoder, ESN) sur ces tâches pour établir une ligne de base de performance.
3. **Analyse des Résultats** : Analyser les performances des différents modèles sur chaque tâche, identifier leurs forces et faiblesses, et proposer des améliorations potentielles.

#### Compétences Développées

Le stagiaire aura l'opportunité de développer plusieurs compétences clés :

- **Programmation en Python** : Utilisation de PyTorch et ReservoirPy pour implémenter et tester les architectures. 
- **Analyse de Données** : Interprétation des résultats des expériences et visualisation des performances des architectures.
- **Recherche et Innovation** : Propositions de nouvelle tâches et améliorations des tâches existantes.
- **Collaboration et Communication** : Travail en équipe avec des chercheurs pour développer et valider les résultats, rédaction de rapports sur les avancées et/ou présentation oral devant l'équipe de recherche.

#### Compétences Requises

- **Connaissances** : Bases solides en Math/Info (des expériences en Machine Learning sont un plus)
- **Programmation** : Bases solides Python (des expériences en Pytorch, Tensorflow, Scikit-Learn sont un plus)
- **Langue** : Une connaissance de l'anglais autant écris que oral est un pré-requis.

#### Ressources et Opportunités

- **Accès à des supercalculateur (JeanZay et/ou Plafrim)** : Cela permettra d'entraîner et tester les modèles les plus performants du moment (Transformers, LSTM, ESN) sur le dataset STREAM, créant ainsi une baseline que d'autres modèles devront surpasser.
- **Participation à un projet Open Source** : L'ensemble du projet sera disponible sur GitHub.
- **Papier de recherche** : Ce projet mènera à la publication d'un papier de recherche où le stagiaire pourra être inclus comme co-auteur. 

#### Équipe & Lieu du stage

Le stage aura lieu à Inria (Bordeaux), sur le NeuroCampus à l'Institut des Maladies Neurodégénératives.
Vous intégrerez l'équipe Mnémosyne du Centre Inria de l'Unversité de Bordeaux.
Plus d'information sur l'équipe : https://team.inria.fr/mnemosyne

#### Encadrants

Vous serez encadrez par Xavier Hinaut (chercheur de l'équipe Mnémosyne de l'Inria) et Yannis Bendi-Ouis (doctorant de Mnémosyne spécialiste des LLM), ce stage contribuera au projet BrainGPT.

Pour toute question ou pour envoyer votre candidature, veuillez contacter Xavier Hinaut à xavier.hinaut@inria.fr et Yannis Bendi-Ouis yannis.bendi-ouis@inria.fr

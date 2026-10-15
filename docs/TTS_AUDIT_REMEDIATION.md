# Traitement de l’audit TTS — 14 octobre 2026

Révision de départ : `ecc93a4335630c3646b4c9e9f1719d4ccde0ac6b`, branche `main`,
arbre propre avant intervention. Travail coordonné avec trois agents GPT-6.1-Sol
en effort low et trois tours de revue indépendante GPT-6-Astra. Les changements
étaient locaux au moment de ce bilan, avant versionnement et sans déploiement.
Qwen reste le moteur par défaut.

## Corrections intégrées

- Capacités Qwen par checkpoint et mode. Une instruction de jeu non prise en
  charge est refusée ; le CustomVoice 1,7B reste un choix explicite.
- QA par alignement des séquences : WER/CER, insertions, suppressions,
  substitutions, transcription brute et alertes de sortie vide/répétition.
  La normalisation conserve chiffres et accents, sans expansion des nombres.
- Concaténation WAV avec contrôle des fréquences et canaux. Le mélange est
  refusé ; l’export séparé utilise un véritable rééchantillonnage FFmpeg.
- Plan de fragments à limites dures, conservation exacte du texte et offsets,
  pauses configurables. Réutilisation du modèle ASR et de ses transcriptions.
- Inférence Qwen dans un processus résident supervisé : seed appliquée dans
  le worker, timeout avec terminate/kill/join, reprise avec un worker neuf,
  libération explicite entre VoiceDesign et clonage. Les prompts transportés
  conservent leur transcription ICL et leurs tenseurs CPU.
- Résolution d’un snapshot complet pour Qwen afin de garder modèle,
  processeur et codec à la même révision. Options de révision et offline.
- Registre : verrou autour des mutations, JSON atomique, import ZIP borné
  à une racine WAV/JSON validée, staging, sauvegarde avant remplacement,
  rollback sur erreur. Aucun pickle importé. Migration additive v2 avec
  dry-run par défaut et sauvegarde lors de l’application.
- Caches locaux vérifiés avant désérialisation : checkpoint, révision, schéma,
  mode, hash de référence/transcription et hash du pickle. Chaque prompt
  garde son propre WAV de référence ; les anciens tons restent utilisables.
  ID complet, préfixe et chemin direct des prompts historiques sont conservés.
- Diagnostics Qwen sur stderr, y compris le probe du processus parent,
  afin de préserver les sorties JSON.
- Adaptateurs facultatifs Pocket/Kokoro, contrat audio commun et imports
  paresseux. Images CPU séparées avec locks/hash de dépendances et base figée.
  Bundles locaux vérifiés, chemins Pocket portables, hashes Kokoro figés.
- Publication des prises CPU sans remplacement d’une prise concurrente.
  WAV brut PCM24 et manifeste séparé ; versions runtime distinguées des
  versions utilisées lors du provisionnement.
- CI Docker sans modèles ni réseau, tests Redis opt-in, export PCM24 distinct
  avec fréquence/canaux explicites. Les étiquettes MIT non justifiées du
  projet ont été retirées ; aucune licence du projet n’a été choisie.

## Vérification et portée des preuves

La suite initiale comptait 77 tests passants dans l’environnement Docker de
vérification. La suite finale, les identifiants d’images, les hashes de sources
et les commandes sont consignés dans
`work/tts-audit-20261014/validation.json`. Les tests unitaires de l’image CI
s’exécutent sans réseau, montage du projet ou poids de modèles. Quatre tests
d’intégration passent avec un Redis jetable sur un réseau Docker isolé.

Kokoro a produit des prises françaises et anglaises dans son image CPU
avec `--network none`. Un pilote compare 1, 2 et 4 threads, en processus froids,
sous quota de 4 CPU et 4 Gio. Les autres activités de développement et l’état du
cache disque ne sont pas contrôlés : ces résultats ne constituent pas une étude
de performance isolée. Les RTF mesurés incluent le chargement ; ils ne prouvent
pas le débit d’un modèle chaud. Aucun score d’écoute n’a été inventé.

Le smoke test Qwen a chargé le Base 0,6B à la révision
`5d83992436eae1d760afd27aff78a71d676296fc`, préparé une référence sinusoïdale
synthétique avec transcription ICL, puis produit un WAV fini valide de 80 ms
à 24 kHz avec un budget de deux tokens. Durées observées : chargement 35,78 s,
conditionnement 9,62 s, génération 28,86 s, total 74,25 s. Deux essais précédents
à douze tokens ont dépassé le délai total de 120 s et leurs workers ont été
arrêtés. Ce test valide le parcours technique ; il ne valide ni le clonage
d’une personne ni la qualité, et n’a pas été exécuté avec un réseau désactivé.

L’image Pocket est construite et ses contrats/relocations sont testés sans
poids. Le provisionnement réel français rencontre un accès Hugging Face
restreint (401) : le propriétaire doit obtenir l’accès aux poids. Aucune
acceptation de conditions n’a été automatisée.

La revue Astra finale donne **GO pour les corrections locales**, sans défaut
bloquant restant dans le périmètre revu. Elle ne vaut ni promotion de moteur,
ni acceptation acoustique, ni clôture de l’ensemble de la feuille de route.

## Utilisation et suites restantes

Voir [les commandes Docker CPU](../apps/voice-cpu/README.md) et
[le protocole de benchmark](TTS_BENCHMARK.md). Le corpus comporte 30 phrases
françaises, 15 anglaises, six paragraphes et deux scripts longs. Le runner
conserve les erreurs/timeouts et produit CSV/JSON, WAV et index d’écoute aveugle.

Les fichiers d’écoute et reçus restent locaux dans `work/tts-audit-20261014/`.
Les actifs Kokoro sont préparés dans `cache/tts/kokoro/`. Les données de voix
existantes n’ont pas été migrées ; aucune expérience SFX n’a été modifiée.

Restent nécessaires pour décider d’un nouveau moteur par défaut : accès et
inférence Pocket, références personnelles autorisées, campagne complète avec
scripts longs, mesure des modèles chauds/mémoire et écoute humaine comparative.
La résidence entre jobs, les reprises ciblées, le mastering avancé et les
candidats secondaires demandent des lots supplémentaires guidés par ces mesures.

L’échange portable actuel transmet une référence principale WAV/JSON et retire
les caches et tons de clonage dérivés. Les références propres à chaque ton sont
conservées localement et dans les sauvegardes de remplacement. Le remplacement
forcé utilise deux renames : un lecteur peut brièvement voir la voix absente ;
une coupure système entre les renames laisse une sauvegarde récupérable. Les
écritures JSON synchronisent le fichier, sans promesse de durabilité du répertoire.
Le planificateur garantit bornes et conservation, pas une analyse linguistique
exhaustive des abréviations ou de la prosodie. Les alignements ASR restent des
horodatages de transcription, pas un alignement forcé du texte approuvé.


Le lot suivant ajoute le batch supervisé et les mesures froid/chaud :
[TTS_PERFORMANCE.md](TTS_PERFORMANCE.md). Les reçus du présent bilan restent le
snapshot du premier lot ; le lot performance dispose de ses propres images,
195 tests et revue Astra.

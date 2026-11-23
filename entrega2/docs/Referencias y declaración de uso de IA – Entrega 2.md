# Referencias y declaración de uso de IA – Entrega 2 (Grupo 36)

Nov 23, 2026&#32;

## Modelos transformer

Estos son los seis modelos preentrenados que se usaron en el proyecto, cada uno con el artículo que lo presenta y su ficha en Hugging Face.

Devlin, J., Chang, M.-W., Lee, K. y Toutanova, K. (2019). BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. En *Proceedings of NAACL-HLT 2019*. [arXiv:1810.04805](https://arxiv.org/abs/1810.04805). Modelo usado en la sección 4.1: [bert-base-multilingual-cased](https://huggingface.co/google-bert/bert-base-multilingual-cased).

Conneau, A., Khandelwal, K., Goyal, N., Chaudhary, V., Wenzek, G., Guzmán, F., Grave, E., Ott, M., Zettlemoyer, L. y Stoyanov, V. (2020). Unsupervised Cross-lingual Representation Learning at Scale. En *Proceedings of ACL 2020*. [arXiv:1911.02116](https://arxiv.org/abs/1911.02116). Modelos usados en las secciones 4.2, 4.5, 4.6, 5 y en el modelo final: [xlm-roberta-base](https://huggingface.co/FacebookAI/xlm-roberta-base) y [xlm-roberta-large](https://huggingface.co/FacebookAI/xlm-roberta-large).

Barbieri, F., Espinosa Anke, L. y Camacho-Collados, J. (2022). XLM-T: Multilingual Language Models in Twitter for Sentiment Analysis and Beyond. En *Proceedings of LREC 2022*. [arXiv:2104.12250](https://arxiv.org/abs/2104.12250). Modelo usado en la sección 4.3: [cardiffnlp/twitter-xlm-roberta-base-sentiment](https://huggingface.co/cardiffnlp/twitter-xlm-roberta-base-sentiment).

He, P., Gao, J. y Chen, W. (2023). DeBERTaV3: Improving DeBERTa using ELECTRA-Style Pre-Training with Gradient-Disentangled Embedding Sharing. En *Proceedings of ICLR 2023*. [arXiv:2111.09543](https://arxiv.org/abs/2111.09543). Modelo usado en las secciones 4.10 y 5.3: [microsoft/mdeberta-v3-base](https://huggingface.co/microsoft/mdeberta-v3-base).

He, P., Liu, X., Gao, J. y Chen, W. (2021). DeBERTa: Decoding-enhanced BERT with Disentangled Attention. En *Proceedings of ICLR 2021*. [arXiv:2006.03654](https://arxiv.org/abs/2006.03654). Arquitectura de mDeBERTa (atención desacoplada), descrita en la sección 4.10.

NLLB Team, Costa-jussà, M. R. et al. (2022). No Language Left Behind: Scaling Human-Centered Machine Translation. *arXiv:2207.04672*. [arXiv:2207.04672](https://arxiv.org/abs/2207.04672). Modelo de traducción usado en la sección 5.3: [facebook/nllb-200-distilled-600M](https://huggingface.co/facebook/nllb-200-distilled-600M).

## Declaración de uso de IA generativa

En este proyecto se usó Claude (Anthropic) como asistente de programación y redacción, a través de Claude Code dentro del editor. El asistente podía leer el notebook y sus salidas, pero no ejecutó los entrenamientos ni las celdas del notebook, ni subió submissions a Kaggle. todo eso lo hizo el autor, y los resultados que aparecen en el notebook provienen de sus ejecuciones.

**Aumentación por traducción (sección 5.3).** El asistente propuso la estrategia de traducir reseñas con NLLB y escribió el código de la traducción, del filtro de consistencia, del equilibrio entre positivas y negativas y de la tabla de evaluación con AUC y umbral elegido por validación cruzada.&#32;

**Análisis de errores.** A partir de preguntas del autor, como cómo atacar los errores en las reseñas mixtas o si los errores opuestos de XLM-RoBERTa base y large se podían aprovechar, el asistente propuso y escribió celdas de diagnóstico y ayudó a interpretar sus salidas.  Al final, el autor no consideró pertinente incluir estos hallazgos pues no aportaban a los modelos.

El asistente no se usó para etiquetar ni revisar a mano las reseñas de evaluación. El autor revisó el código y el texto sugeridos antes de incluirlos, y es el responsable del contenido final del proyecto.

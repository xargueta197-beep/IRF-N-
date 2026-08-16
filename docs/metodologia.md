## Qué mide el IRF-N

El IRF-N estima, cada día, en qué estado del mercado probablemente estamos y
qué tan seguro está el modelo de esa estimación. Por "estado del mercado"
entendemos algo simple: si el mercado está tranquilo (movimientos pequeños,
día tras día parecidos) o agitado (movimientos grandes, impredecibles). A
partir de ahora llamamos "régimen" a cada uno de esos estados.

El indicador no dice si el mercado va a subir o bajar. Dice qué tan agitado
está probablemente, y qué tan seguro está de esa lectura.

## Por qué existe la barra de incertidumbre

El modelo nunca está 100% seguro de en qué régimen estamos: solo ve
probabilidades. Un día puede decir "90% tranquilo, 10% agitado" — bastante
seguro. Otro día puede decir "52% tranquilo, 48% agitado" — casi una
moneda al aire.

Esa segunda situación no es un error del modelo. Es información real: el
mercado está en una zona ambigua donde ni siquiera el modelo puede distinguir
con claridad qué régimen manda. La barra de incertidumbre existe para mostrar
esto sin disfrazarlo. Cuando la barra aparece rayada de arriba a abajo, el
modelo está diciendo, con toda honestidad: "hoy no lo sé con confianza".

Muy pocos indicadores públicos hacen esto. La mayoría fuerza una respuesta
—siempre te dicen "estamos en tal régimen"— aunque por dentro estén casi
adivinando. El IRF-N prefiere decir "no lo sé" cuando es cierto, en vez de
inventar una certeza que no tiene. Esa honestidad es, de hecho, la parte más
importante del indicador: un número seguro pero equivocado es peor que un
número dudoso pero franco.

## La regla más importante — y por qué existe

Imagina un médico que diagnostica con los síntomas que tenía el paciente ese
día. El IRF-N funciona así: cada estimación usa solo la información
disponible en esa fecha.

Existe otra versión del cálculo — llamada "suavizada" — que usa todo el
historial, incluyendo lo que pasó después. Los diagnósticos del pasado se ven
perfectos porque el médico ya sabe cómo terminó todo. Pero ese médico no te
puede ayudar hoy: solo puede explicar el pasado.

El IRF-N publica la versión filtrada. Nunca la suavizada.

Esto importa porque es muy fácil hacer que un modelo "se vea bien" si se le
permite mirar el futuro al calcular el pasado. Un indicador construido así
puede parecer impresionante en una presentación y ser inútil el día que
realmente lo necesitas — porque ese día no tienes el futuro, solo tienes hoy.
Por eso cada número que ves en este panel, para cada fecha, se calculó usando
únicamente lo que se sabía hasta ese día. Ni un dato más.

**El IRF-N publica la probabilidad filtrada. Nunca la suavizada.**

## Qué no puede hacer este indicador

1. **Los estados del mercado son una aproximación.** El mercado no tiene
   estados discretos reales, con una línea que separa "tranquilo" de
   "agitado". El modelo los impone porque es conveniente para calcular, no
   porque sea una verdad del mercado. Un mercado puede moverse de forma
   continua y gradual, y el modelo lo va a clasificar de todos modos en una
   de sus dos categorías, a veces con una confianza que no está del todo
   justificada.

2. **Incertidumbre alta no significa que no haya régimen.** Significa que el
   modelo no lo distingue con los datos disponibles hoy. Son cosas distintas:
   puede que el mercado sí esté claramente en un estado u otro, y que
   simplemente el modelo no tenga todavía evidencia suficiente para verlo.

3. **Resultados históricos no garantizan resultados futuros — y aquí están
   los resultados reales, sin maquillar.** El indicador fue sometido a siete
   pruebas de validación independientes. No todas salieron bien, y se
   reportan igual:

   - **Cuántos regímenes hay:** el criterio estadístico principal (BIC)
     elige dos regímenes de forma clara. Una prueba más exigente (bootstrap
     de Hansen) no tuvo suficiente potencia en esta corrida para confirmarlo
     con un número — quedó pendiente de una corrida más larga.
   - **Calibración:** cuando el modelo dice "70% de probabilidad de régimen
     agitado", acierta con una frecuencia parecida a ese 70% — está bien
     calibrado, incluso un poco conservador. Pero en la métrica de log-loss
     (una forma más estricta de puntuar probabilidades) **no** supera a la
     estrategia ingenua de "predecir siempre la frecuencia histórica de cada
     régimen".
   - **Precisión direccional:** esta prueba (¿acierta el signo del retorno
     mejor que el azar?) no se pudo ejecutar en esta versión por un dato
     faltante en el pipeline de validación. No se inventó un resultado; se
     dejó pendiente.
   - **Comparación con puntos de referencia:** el modelo **sí** le gana de
     forma clara a un modelo de un solo régimen (sin distinguir estados). Pero
     **no** le gana a la estrategia ingenua de "predecir siempre el régimen
     más común".
   - **Desempeño fuera de muestra:** no existe todavía un backtest económico
     (con una regla de inversión real, costos, etc.) — este indicador nunca
     definió una estrategia de trading, a propósito. Lo que sí existe es una
     validación estadística por bloques de tiempo: el modelo aporta valor de
     forma concentrada en los períodos de estrés del mercado (marzo 2020,
     2021, marzo 2023), no de forma pareja en todo momento.
   - **El Sharpe (retorno ajustado por riesgo):** el intervalo de confianza
     del Sharpe condicional al régimen excluye el cero, pero es prácticamente
     idéntico al de simplemente comprar y mantener el activo sin ningún
     modelo. El Sharpe positivo del período pertenece al mercado alcista de
     esos años, no es un aporte demostrado del indicador.
   - **La capa de noticias:** todavía no se pudo evaluar. Faltan datos
     históricos (calendario económico y titulares) para correr esa parte de
     la validación. No se activa en el indicador hasta que se pueda probar.

## Resultados de validación

| Prueba | Resultado |
| :-- | :-- |
| Número de regímenes | Dos regímenes, según el criterio principal (BIC). La prueba más exigente no tuvo suficiente potencia en esta corrida. |
| Calibración | Bien calibrado; gana a la comparación ingenua en una métrica, pierde en otra. |
| Precisión direccional | No se pudo ejecutar en esta versión (dato faltante). |
| Comparación con referencia | Gana a un modelo de un solo régimen; no gana a "predecir siempre el régimen más común". |
| Desempeño fuera de muestra | No hay backtest económico (no existe una regla de inversión). El aporte estadístico se concentra en períodos de estrés. |
| Sharpe (bootstrap) | Distinto de cero, pero casi idéntico al de comprar y mantener sin modelo. |
| Capa de noticias | No evaluable todavía: faltan datos históricos. |

Un indicador que muestra sus propias limitaciones es más confiable que uno
que no.

## Disclaimer

**El IRF-N es un indicador de investigación desarrollado por Araht Analytics.
No constituye recomendación de inversión, asesoría financiera ni promesa de
rendimiento. Los resultados históricos no garantizan resultados futuros.**

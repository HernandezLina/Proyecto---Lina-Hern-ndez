#  Creación de modelo


```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
```

Abrimos la base de datos que anteriormente se contruyo


```python
df = pd.read_csv("ICFES_KVD_1.csv")
df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>COLE_NOMBRE_SEDE-COLE_COD_MCPIO_UBICACION_x</th>
      <th>coordenada_y_sede</th>
      <th>coordenada_x_sede</th>
      <th>Si_KVD</th>
      <th>No_KVD</th>
      <th>estudiantes</th>
      <th>PUNT_GLOBAL</th>
      <th>PUNT_LECTURA_CRITICA</th>
      <th>PUNT_MATEMATICAS</th>
      <th>PUNT_INGLES</th>
      <th>...</th>
      <th>COLE_completa</th>
      <th>COLE_mañana</th>
      <th>COLE_noche</th>
      <th>COLE_sabatina</th>
      <th>COLE_tarde</th>
      <th>COLE_unica</th>
      <th>nse_1</th>
      <th>nse_2</th>
      <th>nse_3</th>
      <th>nse_4</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>INST EDUC FE Y ALEGRIA EL LIMONAR - 5001</td>
      <td>6.17</td>
      <td>-75.64</td>
      <td>0</td>
      <td>1</td>
      <td>47</td>
      <td>222.170213</td>
      <td>50.425532</td>
      <td>44.744681</td>
      <td>42.957447</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.000000</td>
      <td>0.0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>INST EDUC LOMA HERMOSA - 5001</td>
      <td>6.21</td>
      <td>-75.56</td>
      <td>0</td>
      <td>1</td>
      <td>23</td>
      <td>219.000000</td>
      <td>49.782609</td>
      <td>44.695652</td>
      <td>43.913043</td>
      <td>...</td>
      <td>0.0</td>
      <td>1.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>I. E. R. CAMPESTRE NUEVO HORIZONTE - 5148</td>
      <td>6.06</td>
      <td>-75.32</td>
      <td>0</td>
      <td>1</td>
      <td>12</td>
      <td>257.583333</td>
      <td>56.583333</td>
      <td>51.166667</td>
      <td>43.583333</td>
      <td>...</td>
      <td>1.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>I. E. ESCUELA NORMAL SUPERIOR PRESBITERO JOSE ...</td>
      <td>5.71</td>
      <td>-75.31</td>
      <td>0</td>
      <td>1</td>
      <td>22</td>
      <td>275.772727</td>
      <td>57.681818</td>
      <td>57.000000</td>
      <td>51.045455</td>
      <td>...</td>
      <td>0.0</td>
      <td>1.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>C. E. R. ELENA BENITEZ VELEZ - 5847</td>
      <td>6.30</td>
      <td>-76.13</td>
      <td>0</td>
      <td>1</td>
      <td>14</td>
      <td>211.428571</td>
      <td>47.357143</td>
      <td>46.785714</td>
      <td>36.714286</td>
      <td>...</td>
      <td>0.0</td>
      <td>1.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>984</th>
      <td>I.E. PALO ALTO - SEDE PRINCIPAL - 70713</td>
      <td>9.83</td>
      <td>-75.43</td>
      <td>0</td>
      <td>1</td>
      <td>37</td>
      <td>207.405405</td>
      <td>44.243243</td>
      <td>41.297297</td>
      <td>34.891892</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.648649</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.351351</td>
      <td>0.0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>985</th>
      <td>I.E. LA INMACULADA CONCEPCION - SEDE PRINCIPAL...</td>
      <td>9.40</td>
      <td>-75.50</td>
      <td>0</td>
      <td>1</td>
      <td>23</td>
      <td>247.434783</td>
      <td>52.347826</td>
      <td>50.434783</td>
      <td>48.826087</td>
      <td>...</td>
      <td>0.0</td>
      <td>1.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>986</th>
      <td>CONC DE DESARR RUR PLANADAS - 73555</td>
      <td>3.12</td>
      <td>-75.62</td>
      <td>0</td>
      <td>1</td>
      <td>15</td>
      <td>213.000000</td>
      <td>48.466667</td>
      <td>40.866667</td>
      <td>38.066667</td>
      <td>...</td>
      <td>0.0</td>
      <td>1.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>987</th>
      <td>I.E. INTERNADO SANTA TERESITA DEL TUPARRO - SE...</td>
      <td>4.50</td>
      <td>-69.81</td>
      <td>1</td>
      <td>0</td>
      <td>38</td>
      <td>210.815789</td>
      <td>44.078947</td>
      <td>43.631579</td>
      <td>42.157895</td>
      <td>...</td>
      <td>1.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>988</th>
      <td>I.E. INTERNADO TEHODORO WEIJNEN - SEDE PRINCIP...</td>
      <td>5.12</td>
      <td>-70.39</td>
      <td>1</td>
      <td>0</td>
      <td>14</td>
      <td>197.571429</td>
      <td>43.857143</td>
      <td>39.642857</td>
      <td>35.000000</td>
      <td>...</td>
      <td>1.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>989 rows × 45 columns</p>
</div>




```python
df.describe().transpose()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>count</th>
      <th>mean</th>
      <th>std</th>
      <th>min</th>
      <th>25%</th>
      <th>50%</th>
      <th>75%</th>
      <th>max</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>coordenada_y_sede</th>
      <td>989.0</td>
      <td>5.055662</td>
      <td>2.997350</td>
      <td>0.00</td>
      <td>2.630000</td>
      <td>5.000000</td>
      <td>7.100000</td>
      <td>12.580000</td>
    </tr>
    <tr>
      <th>coordenada_x_sede</th>
      <td>989.0</td>
      <td>-70.504257</td>
      <td>18.140849</td>
      <td>-81.71</td>
      <td>-75.900000</td>
      <td>-75.130000</td>
      <td>-73.980000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>Si_KVD</th>
      <td>989.0</td>
      <td>0.317492</td>
      <td>0.465736</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>No_KVD</th>
      <td>989.0</td>
      <td>0.682508</td>
      <td>0.465736</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>estudiantes</th>
      <td>989.0</td>
      <td>18.747219</td>
      <td>11.180467</td>
      <td>3.00</td>
      <td>10.000000</td>
      <td>17.000000</td>
      <td>25.000000</td>
      <td>49.000000</td>
    </tr>
    <tr>
      <th>PUNT_GLOBAL</th>
      <td>989.0</td>
      <td>224.144974</td>
      <td>24.620574</td>
      <td>153.10</td>
      <td>207.405405</td>
      <td>224.447368</td>
      <td>239.913043</td>
      <td>303.461538</td>
    </tr>
    <tr>
      <th>PUNT_LECTURA_CRITICA</th>
      <td>989.0</td>
      <td>47.655239</td>
      <td>4.760273</td>
      <td>31.50</td>
      <td>44.270833</td>
      <td>47.850000</td>
      <td>50.680000</td>
      <td>61.962963</td>
    </tr>
    <tr>
      <th>PUNT_MATEMATICAS</th>
      <td>989.0</td>
      <td>46.278213</td>
      <td>5.911050</td>
      <td>29.00</td>
      <td>42.166667</td>
      <td>46.551020</td>
      <td>50.102041</td>
      <td>65.666667</td>
    </tr>
    <tr>
      <th>PUNT_INGLES</th>
      <td>989.0</td>
      <td>42.243770</td>
      <td>5.322520</td>
      <td>26.00</td>
      <td>38.434783</td>
      <td>42.064516</td>
      <td>45.642857</td>
      <td>60.800000</td>
    </tr>
    <tr>
      <th>F</th>
      <td>989.0</td>
      <td>0.537520</td>
      <td>0.158661</td>
      <td>0.00</td>
      <td>0.450000</td>
      <td>0.545455</td>
      <td>0.636364</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>M</th>
      <td>989.0</td>
      <td>0.462480</td>
      <td>0.158661</td>
      <td>0.00</td>
      <td>0.363636</td>
      <td>0.454545</td>
      <td>0.550000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>Si_etnia</th>
      <td>989.0</td>
      <td>0.167706</td>
      <td>0.340698</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.047619</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>NO_etnia</th>
      <td>989.0</td>
      <td>0.832294</td>
      <td>0.340698</td>
      <td>0.00</td>
      <td>0.952381</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>lec_1</th>
      <td>989.0</td>
      <td>0.152971</td>
      <td>0.115774</td>
      <td>0.00</td>
      <td>0.074074</td>
      <td>0.142857</td>
      <td>0.214286</td>
      <td>0.750000</td>
    </tr>
    <tr>
      <th>lec_2</th>
      <td>989.0</td>
      <td>0.469610</td>
      <td>0.156271</td>
      <td>0.00</td>
      <td>0.371429</td>
      <td>0.470588</td>
      <td>0.571429</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>lec_3</th>
      <td>989.0</td>
      <td>0.262378</td>
      <td>0.135274</td>
      <td>0.00</td>
      <td>0.176471</td>
      <td>0.257143</td>
      <td>0.333333</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>lec_4</th>
      <td>989.0</td>
      <td>0.089744</td>
      <td>0.086611</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.076923</td>
      <td>0.136364</td>
      <td>0.750000</td>
    </tr>
    <tr>
      <th>lec_5</th>
      <td>989.0</td>
      <td>0.025296</td>
      <td>0.047216</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.038462</td>
      <td>0.333333</td>
    </tr>
    <tr>
      <th>int_1</th>
      <td>989.0</td>
      <td>0.135639</td>
      <td>0.125527</td>
      <td>0.00</td>
      <td>0.046512</td>
      <td>0.111111</td>
      <td>0.200000</td>
      <td>0.666667</td>
    </tr>
    <tr>
      <th>int_2</th>
      <td>989.0</td>
      <td>0.279461</td>
      <td>0.137179</td>
      <td>0.00</td>
      <td>0.200000</td>
      <td>0.266667</td>
      <td>0.357143</td>
      <td>0.833333</td>
    </tr>
    <tr>
      <th>int_3</th>
      <td>989.0</td>
      <td>0.339714</td>
      <td>0.146711</td>
      <td>0.00</td>
      <td>0.250000</td>
      <td>0.333333</td>
      <td>0.428571</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>int_4</th>
      <td>989.0</td>
      <td>0.177209</td>
      <td>0.119987</td>
      <td>0.00</td>
      <td>0.100000</td>
      <td>0.166667</td>
      <td>0.250000</td>
      <td>0.666667</td>
    </tr>
    <tr>
      <th>int_5</th>
      <td>989.0</td>
      <td>0.067976</td>
      <td>0.079152</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.050000</td>
      <td>0.111111</td>
      <td>0.500000</td>
    </tr>
    <tr>
      <th>No_oficial</th>
      <td>989.0</td>
      <td>0.002022</td>
      <td>0.044947</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>Oficial</th>
      <td>989.0</td>
      <td>0.997978</td>
      <td>0.044947</td>
      <td>0.00</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>A_Calendario</th>
      <td>989.0</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>1.00</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>B_Calendario</th>
      <td>989.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>Otro_Calendario</th>
      <td>989.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>No_BILINGUE</th>
      <td>989.0</td>
      <td>0.979778</td>
      <td>0.140832</td>
      <td>0.00</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>Si_BILINGUE</th>
      <td>989.0</td>
      <td>0.020222</td>
      <td>0.140832</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>COLE_Academic</th>
      <td>989.0</td>
      <td>0.525412</td>
      <td>0.499371</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>COLE_Tecnic</th>
      <td>989.0</td>
      <td>0.232930</td>
      <td>0.422633</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>COLE_Tec_ac</th>
      <td>989.0</td>
      <td>0.239636</td>
      <td>0.427077</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>COLE_otro_car</th>
      <td>989.0</td>
      <td>0.002022</td>
      <td>0.044947</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>COLE_completa</th>
      <td>989.0</td>
      <td>0.189824</td>
      <td>0.389189</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>COLE_mañana</th>
      <td>989.0</td>
      <td>0.665244</td>
      <td>0.461572</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>COLE_noche</th>
      <td>989.0</td>
      <td>0.007817</td>
      <td>0.066387</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>COLE_sabatina</th>
      <td>989.0</td>
      <td>0.026199</td>
      <td>0.125470</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>COLE_tarde</th>
      <td>989.0</td>
      <td>0.026490</td>
      <td>0.156842</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>COLE_unica</th>
      <td>989.0</td>
      <td>0.084426</td>
      <td>0.275233</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>nse_1</th>
      <td>989.0</td>
      <td>0.434783</td>
      <td>0.495979</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>nse_2</th>
      <td>989.0</td>
      <td>0.553084</td>
      <td>0.497426</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>nse_3</th>
      <td>989.0</td>
      <td>0.012133</td>
      <td>0.109537</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>nse_4</th>
      <td>989.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
  </tbody>
</table>
</div>



Se observa que a nivel general la proporción de niñas presentando la prueba ICFEES Saber 11º es mayor que el de los niños. También, la proporción de estudiantes que pertenecen a una etnia y presentan la prueba es mucho menor (16,97%) que aquellos que la presentan y que no son de un grupo etnico (83,02%). Podemos ver que la mayoría de los estudiantes leen entre 30 minutos o menos por entretenimiento, mientras que navegan en internet entre 30 minutos y una hora. En cuento a características de la sede educativa, todas las observaciones son calendario A, más del 98% no es bilingüe, el 55% son de carácter académico, el 66% tiene jornada por la mañana y más del 98% es de un nivel socioeconómico muy bajo o bajo.


```python
#Revisar columnas
data_columns = list(df.columns)
a = data_columns [3:]
a
```




    ['Si_KVD',
     'No_KVD',
     'estudiantes',
     'PUNT_GLOBAL',
     'PUNT_LECTURA_CRITICA',
     'PUNT_MATEMATICAS',
     'PUNT_INGLES',
     'F',
     'M',
     'Si_etnia',
     'NO_etnia',
     'lec_1',
     'lec_2',
     'lec_3',
     'lec_4',
     'lec_5',
     'int_1',
     'int_2',
     'int_3',
     'int_4',
     'int_5',
     'No_oficial',
     'Oficial',
     'A_Calendario',
     'B_Calendario',
     'Otro_Calendario',
     'No_BILINGUE',
     'Si_BILINGUE',
     'COLE_Academic',
     'COLE_Tecnic',
     'COLE_Tec_ac',
     'COLE_otro_car',
     'COLE_completa',
     'COLE_mañana',
     'COLE_noche',
     'COLE_sabatina',
     'COLE_tarde',
     'COLE_unica',
     'nse_1',
     'nse_2',
     'nse_3',
     'nse_4']




```python
#Columnas de interes para hacer un mapa de calor con las correlaciones
data_columns = list(df.columns)
a = data_columns [3:15]
interest = df[a]
correlaciones = interest.corr()
#matriz 
plt.figure(figsize=(10, 10))
sns.heatmap(correlaciones, cmap='Greens', annot = True)
plt.title ('Matriz de correlación')
plt.show();
```


    
![png](output_7_0.png)
    



```python
#Columnas de interes para hacer un mapa de calor con las correlaciones
data_columns = list(df.columns)
a = data_columns [6:10]
b = data_columns [16:26]
c = data_columns [29:34]
interes = a + b + c
columns = df[interes]
correlaciones = columns.corr()

#matriz 
plt.figure(figsize=(15, 15))
sns.heatmap(correlaciones, cmap='Greens', annot = True)
plt.title ('Matriz de correlación')
plt.show();
```


    
![png](output_8_0.png)
    



```python
#Columnas de interes para hacer un mapa de calor con las correlaciones
data_columns = list(df.columns)
a = data_columns [6:10]
b = data_columns [34:]
interes = a + b 
columns = df[interes]
correlaciones = columns.corr()

#matriz 
plt.figure(figsize=(15, 15))
sns.heatmap(correlaciones, cmap='Greens', annot = True)
plt.title ('Matriz de correlación')
plt.show();
```


    
![png](output_9_0.png)
    


A partir de los anteriores mapas de calor se evidencian correlaciones de mas de 0.1 con genero, etnia, lectura entre 30 minutos y una hora, no navegación en internet, caracter bilingüe,  caracter academico y nivel socioeconomico.


```python
df.to_csv('~/Downloads/final.csv', index = False)
#Se guarda la base final para utilizarla en tableaub
```


```python
df.columns
```




    Index(['COLE_NOMBRE_SEDE-COLE_COD_MCPIO_UBICACION_x', 'coordenada_y_sede',
           'coordenada_x_sede', 'Si_KVD', 'No_KVD', 'estudiantes', 'PUNT_GLOBAL',
           'PUNT_LECTURA_CRITICA', 'PUNT_MATEMATICAS', 'PUNT_INGLES', 'F', 'M',
           'Si_etnia', 'NO_etnia', 'lec_1', 'lec_2', 'lec_3', 'lec_4', 'lec_5',
           'int_1', 'int_2', 'int_3', 'int_4', 'int_5', 'No_oficial', 'Oficial',
           'A_Calendario', 'B_Calendario', 'Otro_Calendario', 'No_BILINGUE',
           'Si_BILINGUE', 'COLE_Academic', 'COLE_Tecnic', 'COLE_Tec_ac',
           'COLE_otro_car', 'COLE_completa', 'COLE_mañana', 'COLE_noche',
           'COLE_sabatina', 'COLE_tarde', 'COLE_unica', 'nse_1', 'nse_2', 'nse_3',
           'nse_4'],
          dtype='object')




```python
#Puntaje Global medio = a + B(...) 
model = smf.ols(formula = 'PUNT_GLOBAL ~ Si_KVD + F + Si_etnia + lec_3 + int_1 + A_Calendario + COLE_noche + nse_3',  data=df).fit()
model.summary()
```




<table class="simpletable">
<caption>OLS Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>       <td>PUNT_GLOBAL</td>   <th>  R-squared:         </th> <td>   0.218</td>
</tr>
<tr>
  <th>Model:</th>                   <td>OLS</td>       <th>  Adj. R-squared:    </th> <td>   0.212</td>
</tr>
<tr>
  <th>Method:</th>             <td>Least Squares</td>  <th>  F-statistic:       </th> <td>   38.97</td>
</tr>
<tr>
  <th>Date:</th>             <td>Wed, 24 Nov 2021</td> <th>  Prob (F-statistic):</th> <td>1.97e-48</td>
</tr>
<tr>
  <th>Time:</th>                 <td>00:00:02</td>     <th>  Log-Likelihood:    </th> <td> -4449.8</td>
</tr>
<tr>
  <th>No. Observations:</th>      <td>   989</td>      <th>  AIC:               </th> <td>   8916.</td>
</tr>
<tr>
  <th>Df Residuals:</th>          <td>   981</td>      <th>  BIC:               </th> <td>   8955.</td>
</tr>
<tr>
  <th>Df Model:</th>              <td>     7</td>      <th>                     </th>     <td> </td>   
</tr>
<tr>
  <th>Covariance Type:</th>      <td>nonrobust</td>    <th>                     </th>     <td> </td>   
</tr>
</table>
<table class="simpletable">
<tr>
        <td></td>          <th>coef</th>     <th>std err</th>      <th>t</th>      <th>P>|t|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>Intercept</th>    <td>  109.4712</td> <td>    1.481</td> <td>   73.917</td> <td> 0.000</td> <td>  106.565</td> <td>  112.378</td>
</tr>
<tr>
  <th>Si_KVD</th>       <td>   -3.2966</td> <td>    1.509</td> <td>   -2.185</td> <td> 0.029</td> <td>   -6.257</td> <td>   -0.336</td>
</tr>
<tr>
  <th>F</th>            <td>   17.4867</td> <td>    4.399</td> <td>    3.975</td> <td> 0.000</td> <td>    8.854</td> <td>   26.119</td>
</tr>
<tr>
  <th>Si_etnia</th>     <td>  -25.5871</td> <td>    2.057</td> <td>  -12.439</td> <td> 0.000</td> <td>  -29.624</td> <td>  -21.550</td>
</tr>
<tr>
  <th>lec_3</th>        <td>   17.3816</td> <td>    5.169</td> <td>    3.362</td> <td> 0.001</td> <td>    7.237</td> <td>   27.526</td>
</tr>
<tr>
  <th>int_1</th>        <td>  -26.5318</td> <td>    5.610</td> <td>   -4.730</td> <td> 0.000</td> <td>  -37.540</td> <td>  -15.524</td>
</tr>
<tr>
  <th>A_Calendario</th> <td>  109.4712</td> <td>    1.481</td> <td>   73.917</td> <td> 0.000</td> <td>  106.565</td> <td>  112.378</td>
</tr>
<tr>
  <th>COLE_noche</th>   <td>  -22.5752</td> <td>   10.526</td> <td>   -2.145</td> <td> 0.032</td> <td>  -43.231</td> <td>   -1.920</td>
</tr>
<tr>
  <th>nse_3</th>        <td>   29.2956</td> <td>    6.387</td> <td>    4.587</td> <td> 0.000</td> <td>   16.762</td> <td>   41.830</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Omnibus:</th>       <td> 5.556</td> <th>  Durbin-Watson:     </th> <td>   1.317</td>
</tr>
<tr>
  <th>Prob(Omnibus):</th> <td> 0.062</td> <th>  Jarque-Bera (JB):  </th> <td>   5.639</td>
</tr>
<tr>
  <th>Skew:</th>          <td> 0.181</td> <th>  Prob(JB):          </th> <td>  0.0596</td>
</tr>
<tr>
  <th>Kurtosis:</th>      <td> 2.928</td> <th>  Cond. No.          </th> <td>4.62e+15</td>
</tr>
</table><br/><br/>Notes:<br/>[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.<br/>[2] The smallest eigenvalue is 1.17e-28. This might indicate that there are<br/>strong multicollinearity problems or that the design matrix is singular.



Esta regresión es el resultado de varias iteraciones en las que se eliminaron aquellas variables que presentaban un p-value muy alto. Esta regresión se hace con un nivel de significancia del 5% y presenta un valor de ajuste bajo de 0.218 esto muestra que el modelo solo explica alrededor del 21% del puntaje medio global de las sedes educativas pequeñas rurales. Esto se debe mayoritariamente a la existencia de variables omitidas que por cuestiones de tiempo no pudieron ser recaudadas. 

Se presenta el calendario, la etnia, la no navegación en internet, la jornada y el nivel socio económico como las variables con una correlación mayor con el puntaje global. Sin embargo, es importante mencionar que los kioscos se asocian con una disminución de 3,29 puntos en el puntaje medio global, esto es significativo y contrario a lo que se esperaba. 

A forma de conclusión, y al ver el efecto negativo de los KVD en el puntaje global medio de las sedes educativas es necesario continuar con la investigación buscando explicación a este efecto. Una hipótesis respaldada por O’Dwyer et al (2005) es que, el efecto positivo de la introducción de tecnología en la educación de niños, niñas y jóvenes es explicada por la apropiación de docentes en las sedes educativas y el buen manejo de estos más que su simple introducción. 

Esto, podría verificarse a partir de una prueba de diferencias en diferencias con el fin de controlar también efectos de rendimiento académico anteriores a la introducción de los KVD y también la forma en que profesores y directivos incluyen estos en sus clases o metodologías educativas. 

Igualmente, es importante incluir variables a nivel local como la existencia de redes eléctricas o la clase de conexión (fibra óptica, cable coaxial, satelital o radioenlaces) ya que esto juega un papel fundamental en el funcionamiento adecuado del KVD.

# Base final


```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
```


```python
#Abrimos la base de la parte dos
b = pd.read_csv("completa.csv")
```


```python
b.shape
```




    (2344, 56)




```python
b
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
      <th>COLE_COD_DANE_SEDE</th>
      <th>COLE_NOMBRE_SEDE</th>
      <th>COLE_COD_MCPIO_UBICACION</th>
      <th>estudiantes</th>
      <th>PUNT_GLOBAL</th>
      <th>PUNT_LECTURA_CRITICA</th>
      <th>PUNT_MATEMATICAS</th>
      <th>PUNT_INGLES</th>
      <th>F</th>
      <th>M</th>
      <th>...</th>
      <th>nombre_sede_x</th>
      <th>tipo_kvd</th>
      <th>estado_del_kiosco</th>
      <th>COLE_NOMBRE_SEDE-COLE_COD_MCPIO_UBICACION_x</th>
      <th>cod_dane_municipio</th>
      <th>codigo_dane_sede</th>
      <th>coordenada_y_sede</th>
      <th>coordenada_x_sede</th>
      <th>nombre_sede_y</th>
      <th>COLE_NOMBRE_SEDE-COLE_COD_MCPIO_UBICACION_y</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1,05001</td>
      <td>INST EDUC FE Y ALEGRIA EL LIMONAR</td>
      <td>5001</td>
      <td>47</td>
      <td>222.170213</td>
      <td>50.425532</td>
      <td>44.744681</td>
      <td>42.957447</td>
      <td>0.553191</td>
      <td>0.446809</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>INST EDUC FE Y ALEGRIA EL LIMONAR - 5001</td>
      <td>5001.0</td>
      <td>1.050010e+11</td>
      <td>6.17</td>
      <td>-75.64</td>
      <td>INST EDUC FE Y ALEGRIA EL LIMONAR</td>
      <td>INST EDUC FE Y ALEGRIA EL LIMONAR - 05001</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1,05001</td>
      <td>INST EDUC LOMA HERMOSA</td>
      <td>5001</td>
      <td>23</td>
      <td>219.000000</td>
      <td>49.782609</td>
      <td>44.695652</td>
      <td>43.913043</td>
      <td>0.521739</td>
      <td>0.478261</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>INST EDUC LOMA HERMOSA - 5001</td>
      <td>5001.0</td>
      <td>1.050010e+11</td>
      <td>6.21</td>
      <td>-75.56</td>
      <td>INST EDUC LOMA HERMOSA</td>
      <td>INST EDUC LOMA HERMOSA - 05001</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1,05148</td>
      <td>I. E. R. CAMPESTRE NUEVO HORIZONTE</td>
      <td>5148</td>
      <td>12</td>
      <td>257.583333</td>
      <td>56.583333</td>
      <td>51.166667</td>
      <td>43.583333</td>
      <td>0.750000</td>
      <td>0.250000</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>I. E. R. CAMPESTRE NUEVO HORIZONTE - 5148</td>
      <td>5148.0</td>
      <td>1.051480e+11</td>
      <td>6.06</td>
      <td>-75.32</td>
      <td>I. E. R. CAMPESTRE NUEVO HORIZONTE</td>
      <td>I. E. R. CAMPESTRE NUEVO HORIZONTE - 05148</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1,0544</td>
      <td>INSTITUTO TECNICO DE MARINILLA</td>
      <td>5440</td>
      <td>23</td>
      <td>268.782609</td>
      <td>55.130435</td>
      <td>58.130435</td>
      <td>48.739130</td>
      <td>0.478261</td>
      <td>0.521739</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>INSTITUTO TECNICO DE MARINILLA - 5440</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1,05679</td>
      <td>LICEO TOMAS EASTMAN</td>
      <td>5679</td>
      <td>115</td>
      <td>237.478261</td>
      <td>51.095652</td>
      <td>48.330435</td>
      <td>44.460870</td>
      <td>0.495652</td>
      <td>0.504348</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>LICEO TOMAS EASTMAN - 5679</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
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
      <th>2339</th>
      <td>4,85001</td>
      <td>GIMNASIO DE LOS LLANOS</td>
      <td>85001</td>
      <td>53</td>
      <td>326.150943</td>
      <td>65.226415</td>
      <td>67.188679</td>
      <td>70.339623</td>
      <td>0.396226</td>
      <td>0.603774</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>GIMNASIO DE LOS LLANOS - 85001</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2340</th>
      <td>4,86001</td>
      <td>I.E.R. MAYOYOQUE - SEDE PRINCIPAL</td>
      <td>86571</td>
      <td>12</td>
      <td>215.166667</td>
      <td>45.083333</td>
      <td>42.500000</td>
      <td>39.083333</td>
      <td>0.500000</td>
      <td>0.500000</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>I.E.R. MAYOYOQUE - SEDE PRINCIPAL - 86571</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2341</th>
      <td>4,86865</td>
      <td>I.E.R. SAN MARCELINO - SEDE PRINCIPAL</td>
      <td>86757</td>
      <td>8</td>
      <td>181.125000</td>
      <td>38.125000</td>
      <td>35.875000</td>
      <td>31.250000</td>
      <td>0.750000</td>
      <td>0.250000</td>
      <td>...</td>
      <td>I.E.R. SAN MARCELINO - SEDE PRINCIPAL</td>
      <td>SEDE EDUCATIVA</td>
      <td>EN OPERACION</td>
      <td>I.E.R. SAN MARCELINO - SEDE PRINCIPAL - 86757</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2342</th>
      <td>4,99001</td>
      <td>I.E. INTERNADO SANTA TERESITA DEL TUPARRO - SE...</td>
      <td>99773</td>
      <td>38</td>
      <td>210.815789</td>
      <td>44.078947</td>
      <td>43.631579</td>
      <td>42.157895</td>
      <td>0.315789</td>
      <td>0.684211</td>
      <td>...</td>
      <td>I.E. INTERNADO SANTA TERESITA DEL TUPARRO - SE...</td>
      <td>SEDE EDUCATIVA</td>
      <td>EN OPERACION</td>
      <td>I.E. INTERNADO SANTA TERESITA DEL TUPARRO - SE...</td>
      <td>99773.0</td>
      <td>4.990010e+11</td>
      <td>4.50</td>
      <td>-69.81</td>
      <td>I.E. INTERNADO SANTA TERESITA DEL TUPARRO - SE...</td>
      <td>I.E. INTERNADO SANTA TERESITA DEL TUPARRO - SE...</td>
    </tr>
    <tr>
      <th>2343</th>
      <td>4,99001</td>
      <td>I.E. INTERNADO TEHODORO WEIJNEN - SEDE PRINCIPAL</td>
      <td>99524</td>
      <td>14</td>
      <td>197.571429</td>
      <td>43.857143</td>
      <td>39.642857</td>
      <td>35.000000</td>
      <td>0.214286</td>
      <td>0.785714</td>
      <td>...</td>
      <td>I.E. INTERNADO TEHODORO WEIJNEN - SEDE PRINCIPAL</td>
      <td>SEDE EDUCATIVA</td>
      <td>EN OPERACION</td>
      <td>I.E. INTERNADO TEHODORO WEIJNEN - SEDE PRINCIP...</td>
      <td>99524.0</td>
      <td>4.990010e+11</td>
      <td>5.12</td>
      <td>-70.39</td>
      <td>I.E. INTERNADO TEHODORO WEIJNEN - SEDE PRINCIPAL</td>
      <td>I.E. INTERNADO TEHODORO WEIJNEN - SEDE PRINCIP...</td>
    </tr>
  </tbody>
</table>
<p>2344 rows × 56 columns</p>
</div>




```python
b.isna().sum()
```




    COLE_COD_DANE_SEDE                                0
    COLE_NOMBRE_SEDE                                  0
    COLE_COD_MCPIO_UBICACION                          0
    estudiantes                                       0
    PUNT_GLOBAL                                       0
    PUNT_LECTURA_CRITICA                              0
    PUNT_MATEMATICAS                                  0
    PUNT_INGLES                                       0
    F                                                 0
    M                                                 0
    Si_etnia                                          0
    NO_etnia                                          0
    lec_1                                             0
    lec_2                                             0
    lec_3                                             0
    lec_4                                             0
    lec_5                                             0
    int_1                                             0
    int_2                                             0
    int_3                                             0
    int_4                                             0
    int_5                                             0
    No_oficial                                        0
    Oficial                                           0
    A_Calendario                                      0
    B_Calendario                                      0
    Otro_Calendario                                   0
    No_BILINGUE                                       0
    Si_BILINGUE                                       0
    COLE_Academic                                     0
    COLE_Tecnic                                       0
    COLE_Tec_ac                                       0
    COLE_otro_car                                     0
    COLE_completa                                     0
    COLE_mañana                                       0
    COLE_noche                                        0
    COLE_sabatina                                     0
    COLE_tarde                                        0
    COLE_unica                                        0
    nse_1                                             0
    nse_2                                             0
    nse_3                                             0
    nse_4                                             0
    dane_municipio                                 1746
    dane_institucion_educativa                     1746
    dane_sede                                      1746
    nombre_sede_x                                  1746
    tipo_kvd                                       1746
    estado_del_kiosco                              1746
    COLE_NOMBRE_SEDE-COLE_COD_MCPIO_UBICACION_x       0
    cod_dane_municipio                             1177
    codigo_dane_sede                               1177
    coordenada_y_sede                              1183
    coordenada_x_sede                              1183
    nombre_sede_y                                  1177
    COLE_NOMBRE_SEDE-COLE_COD_MCPIO_UBICACION_y    1177
    dtype: int64



Puesto que es de interes el ubicar las sedes educativas, se eliminan aquellas observaciones sin información en sus coordenadas


```python
df = b[b['coordenada_y_sede'].notna()]
#De esta menra la muestra se reduce a 1.161 sedes educativas
```


```python
df.isna().sum()
```




    COLE_COD_DANE_SEDE                               0
    COLE_NOMBRE_SEDE                                 0
    COLE_COD_MCPIO_UBICACION                         0
    estudiantes                                      0
    PUNT_GLOBAL                                      0
    PUNT_LECTURA_CRITICA                             0
    PUNT_MATEMATICAS                                 0
    PUNT_INGLES                                      0
    F                                                0
    M                                                0
    Si_etnia                                         0
    NO_etnia                                         0
    lec_1                                            0
    lec_2                                            0
    lec_3                                            0
    lec_4                                            0
    lec_5                                            0
    int_1                                            0
    int_2                                            0
    int_3                                            0
    int_4                                            0
    int_5                                            0
    No_oficial                                       0
    Oficial                                          0
    A_Calendario                                     0
    B_Calendario                                     0
    Otro_Calendario                                  0
    No_BILINGUE                                      0
    Si_BILINGUE                                      0
    COLE_Academic                                    0
    COLE_Tecnic                                      0
    COLE_Tec_ac                                      0
    COLE_otro_car                                    0
    COLE_completa                                    0
    COLE_mañana                                      0
    COLE_noche                                       0
    COLE_sabatina                                    0
    COLE_tarde                                       0
    COLE_unica                                       0
    nse_1                                            0
    nse_2                                            0
    nse_3                                            0
    nse_4                                            0
    dane_municipio                                 812
    dane_institucion_educativa                     812
    dane_sede                                      812
    nombre_sede_x                                  812
    tipo_kvd                                       812
    estado_del_kiosco                              812
    COLE_NOMBRE_SEDE-COLE_COD_MCPIO_UBICACION_x      0
    cod_dane_municipio                               0
    codigo_dane_sede                                 0
    coordenada_y_sede                                0
    coordenada_x_sede                                0
    nombre_sede_y                                    0
    COLE_NOMBRE_SEDE-COLE_COD_MCPIO_UBICACION_y      0
    dtype: int64




```python
#Cambiar tipo_kvd por dummy
df['Si_KVD'] = np.where(df['tipo_kvd'] == 'SEDE EDUCATIVA', 1, 0)
df['No_KVD'] = np.where(df['tipo_kvd'] != 'SEDE EDUCATIVA', 1, 0)
```

    <ipython-input-8-3ded975508a3>:2: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      df['Si_KVD'] = np.where(df['tipo_kvd'] == 'SEDE EDUCATIVA', 1, 0)
    <ipython-input-8-3ded975508a3>:3: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      df['No_KVD'] = np.where(df['tipo_kvd'] != 'SEDE EDUCATIVA', 1, 0)



```python
#Se realiza un ultimo filtro sobre las variables que nos interesan
pd.DataFrame(list(df.columns))
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
      <th>0</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>COLE_COD_DANE_SEDE</td>
    </tr>
    <tr>
      <th>1</th>
      <td>COLE_NOMBRE_SEDE</td>
    </tr>
    <tr>
      <th>2</th>
      <td>COLE_COD_MCPIO_UBICACION</td>
    </tr>
    <tr>
      <th>3</th>
      <td>estudiantes</td>
    </tr>
    <tr>
      <th>4</th>
      <td>PUNT_GLOBAL</td>
    </tr>
    <tr>
      <th>5</th>
      <td>PUNT_LECTURA_CRITICA</td>
    </tr>
    <tr>
      <th>6</th>
      <td>PUNT_MATEMATICAS</td>
    </tr>
    <tr>
      <th>7</th>
      <td>PUNT_INGLES</td>
    </tr>
    <tr>
      <th>8</th>
      <td>F</td>
    </tr>
    <tr>
      <th>9</th>
      <td>M</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Si_etnia</td>
    </tr>
    <tr>
      <th>11</th>
      <td>NO_etnia</td>
    </tr>
    <tr>
      <th>12</th>
      <td>lec_1</td>
    </tr>
    <tr>
      <th>13</th>
      <td>lec_2</td>
    </tr>
    <tr>
      <th>14</th>
      <td>lec_3</td>
    </tr>
    <tr>
      <th>15</th>
      <td>lec_4</td>
    </tr>
    <tr>
      <th>16</th>
      <td>lec_5</td>
    </tr>
    <tr>
      <th>17</th>
      <td>int_1</td>
    </tr>
    <tr>
      <th>18</th>
      <td>int_2</td>
    </tr>
    <tr>
      <th>19</th>
      <td>int_3</td>
    </tr>
    <tr>
      <th>20</th>
      <td>int_4</td>
    </tr>
    <tr>
      <th>21</th>
      <td>int_5</td>
    </tr>
    <tr>
      <th>22</th>
      <td>No_oficial</td>
    </tr>
    <tr>
      <th>23</th>
      <td>Oficial</td>
    </tr>
    <tr>
      <th>24</th>
      <td>A_Calendario</td>
    </tr>
    <tr>
      <th>25</th>
      <td>B_Calendario</td>
    </tr>
    <tr>
      <th>26</th>
      <td>Otro_Calendario</td>
    </tr>
    <tr>
      <th>27</th>
      <td>No_BILINGUE</td>
    </tr>
    <tr>
      <th>28</th>
      <td>Si_BILINGUE</td>
    </tr>
    <tr>
      <th>29</th>
      <td>COLE_Academic</td>
    </tr>
    <tr>
      <th>30</th>
      <td>COLE_Tecnic</td>
    </tr>
    <tr>
      <th>31</th>
      <td>COLE_Tec_ac</td>
    </tr>
    <tr>
      <th>32</th>
      <td>COLE_otro_car</td>
    </tr>
    <tr>
      <th>33</th>
      <td>COLE_completa</td>
    </tr>
    <tr>
      <th>34</th>
      <td>COLE_mañana</td>
    </tr>
    <tr>
      <th>35</th>
      <td>COLE_noche</td>
    </tr>
    <tr>
      <th>36</th>
      <td>COLE_sabatina</td>
    </tr>
    <tr>
      <th>37</th>
      <td>COLE_tarde</td>
    </tr>
    <tr>
      <th>38</th>
      <td>COLE_unica</td>
    </tr>
    <tr>
      <th>39</th>
      <td>nse_1</td>
    </tr>
    <tr>
      <th>40</th>
      <td>nse_2</td>
    </tr>
    <tr>
      <th>41</th>
      <td>nse_3</td>
    </tr>
    <tr>
      <th>42</th>
      <td>nse_4</td>
    </tr>
    <tr>
      <th>43</th>
      <td>dane_municipio</td>
    </tr>
    <tr>
      <th>44</th>
      <td>dane_institucion_educativa</td>
    </tr>
    <tr>
      <th>45</th>
      <td>dane_sede</td>
    </tr>
    <tr>
      <th>46</th>
      <td>nombre_sede_x</td>
    </tr>
    <tr>
      <th>47</th>
      <td>tipo_kvd</td>
    </tr>
    <tr>
      <th>48</th>
      <td>estado_del_kiosco</td>
    </tr>
    <tr>
      <th>49</th>
      <td>COLE_NOMBRE_SEDE-COLE_COD_MCPIO_UBICACION_x</td>
    </tr>
    <tr>
      <th>50</th>
      <td>cod_dane_municipio</td>
    </tr>
    <tr>
      <th>51</th>
      <td>codigo_dane_sede</td>
    </tr>
    <tr>
      <th>52</th>
      <td>coordenada_y_sede</td>
    </tr>
    <tr>
      <th>53</th>
      <td>coordenada_x_sede</td>
    </tr>
    <tr>
      <th>54</th>
      <td>nombre_sede_y</td>
    </tr>
    <tr>
      <th>55</th>
      <td>COLE_NOMBRE_SEDE-COLE_COD_MCPIO_UBICACION_y</td>
    </tr>
    <tr>
      <th>56</th>
      <td>Si_KVD</td>
    </tr>
    <tr>
      <th>57</th>
      <td>No_KVD</td>
    </tr>
  </tbody>
</table>
</div>




```python
#Se Realiza un ultimo filtro de las las columnas de interes
df_f = df.iloc[:,[49,52,53,56,57,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42]]
df_f.shape
```




    (1161, 45)




```python
df_f
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
      <th>5</th>
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
      <th>6</th>
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
      <th>2322</th>
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
      <th>2323</th>
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
      <th>2325</th>
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
      <th>2342</th>
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
      <th>2343</th>
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
<p>1161 rows × 45 columns</p>
</div>



# Exploración de datos


```python
df_f.describe().transpose()
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
      <td>1161.0</td>
      <td>5.202550</td>
      <td>3.051017</td>
      <td>0.00</td>
      <td>2.780000</td>
      <td>5.080000</td>
      <td>7.550000</td>
      <td>13.370000</td>
    </tr>
    <tr>
      <th>coordenada_x_sede</th>
      <td>1161.0</td>
      <td>-70.760879</td>
      <td>17.714021</td>
      <td>-81.72</td>
      <td>-75.880000</td>
      <td>-75.190000</td>
      <td>-74.010000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>Si_KVD</th>
      <td>1161.0</td>
      <td>0.300603</td>
      <td>0.458718</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>No_KVD</th>
      <td>1161.0</td>
      <td>0.699397</td>
      <td>0.458718</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>estudiantes</th>
      <td>1161.0</td>
      <td>25.055986</td>
      <td>24.470978</td>
      <td>1.00</td>
      <td>10.000000</td>
      <td>18.000000</td>
      <td>31.000000</td>
      <td>253.000000</td>
    </tr>
    <tr>
      <th>PUNT_GLOBAL</th>
      <td>1161.0</td>
      <td>223.980904</td>
      <td>24.879478</td>
      <td>153.10</td>
      <td>207.166667</td>
      <td>224.142857</td>
      <td>240.117647</td>
      <td>303.461538</td>
    </tr>
    <tr>
      <th>PUNT_LECTURA_CRITICA</th>
      <td>1161.0</td>
      <td>47.672144</td>
      <td>4.847121</td>
      <td>31.50</td>
      <td>44.333333</td>
      <td>47.928571</td>
      <td>50.813953</td>
      <td>61.962963</td>
    </tr>
    <tr>
      <th>PUNT_MATEMATICAS</th>
      <td>1161.0</td>
      <td>46.229992</td>
      <td>5.958203</td>
      <td>29.00</td>
      <td>42.038462</td>
      <td>46.444444</td>
      <td>50.102041</td>
      <td>65.666667</td>
    </tr>
    <tr>
      <th>PUNT_INGLES</th>
      <td>1161.0</td>
      <td>42.258849</td>
      <td>5.495655</td>
      <td>24.50</td>
      <td>38.434783</td>
      <td>42.100000</td>
      <td>45.697674</td>
      <td>61.793103</td>
    </tr>
    <tr>
      <th>F</th>
      <td>1161.0</td>
      <td>0.538987</td>
      <td>0.167405</td>
      <td>0.00</td>
      <td>0.461538</td>
      <td>0.550000</td>
      <td>0.636364</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>M</th>
      <td>1161.0</td>
      <td>0.461013</td>
      <td>0.167405</td>
      <td>0.00</td>
      <td>0.363636</td>
      <td>0.450000</td>
      <td>0.538462</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>Si_etnia</th>
      <td>1161.0</td>
      <td>0.173727</td>
      <td>0.347706</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.052632</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>NO_etnia</th>
      <td>1161.0</td>
      <td>0.826273</td>
      <td>0.347706</td>
      <td>0.00</td>
      <td>0.947368</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>lec_1</th>
      <td>1161.0</td>
      <td>0.156174</td>
      <td>0.126668</td>
      <td>0.00</td>
      <td>0.076923</td>
      <td>0.142857</td>
      <td>0.212766</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>lec_2</th>
      <td>1161.0</td>
      <td>0.468376</td>
      <td>0.165789</td>
      <td>0.00</td>
      <td>0.375000</td>
      <td>0.468750</td>
      <td>0.562500</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>lec_3</th>
      <td>1161.0</td>
      <td>0.259116</td>
      <td>0.137133</td>
      <td>0.00</td>
      <td>0.176471</td>
      <td>0.256410</td>
      <td>0.333333</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>lec_4</th>
      <td>1161.0</td>
      <td>0.091545</td>
      <td>0.091746</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.080000</td>
      <td>0.133333</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>lec_5</th>
      <td>1161.0</td>
      <td>0.024789</td>
      <td>0.044599</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.037037</td>
      <td>0.333333</td>
    </tr>
    <tr>
      <th>int_1</th>
      <td>1161.0</td>
      <td>0.133186</td>
      <td>0.134469</td>
      <td>0.00</td>
      <td>0.045455</td>
      <td>0.100000</td>
      <td>0.183099</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>int_2</th>
      <td>1161.0</td>
      <td>0.274333</td>
      <td>0.142548</td>
      <td>0.00</td>
      <td>0.190476</td>
      <td>0.259259</td>
      <td>0.344828</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>int_3</th>
      <td>1161.0</td>
      <td>0.339357</td>
      <td>0.152779</td>
      <td>0.00</td>
      <td>0.250000</td>
      <td>0.339623</td>
      <td>0.423077</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>int_4</th>
      <td>1161.0</td>
      <td>0.181087</td>
      <td>0.125035</td>
      <td>0.00</td>
      <td>0.100000</td>
      <td>0.173913</td>
      <td>0.250000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>int_5</th>
      <td>1161.0</td>
      <td>0.072038</td>
      <td>0.080953</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.052632</td>
      <td>0.115385</td>
      <td>0.500000</td>
    </tr>
    <tr>
      <th>No_oficial</th>
      <td>1161.0</td>
      <td>0.001723</td>
      <td>0.041487</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>Oficial</th>
      <td>1161.0</td>
      <td>0.998277</td>
      <td>0.041487</td>
      <td>0.00</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>A_Calendario</th>
      <td>1161.0</td>
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
      <td>1161.0</td>
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
      <td>1161.0</td>
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
      <td>1161.0</td>
      <td>0.981051</td>
      <td>0.136404</td>
      <td>0.00</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>Si_BILINGUE</th>
      <td>1161.0</td>
      <td>0.018949</td>
      <td>0.136404</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>COLE_Academic</th>
      <td>1161.0</td>
      <td>0.509589</td>
      <td>0.499923</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>COLE_Tecnic</th>
      <td>1161.0</td>
      <td>0.225984</td>
      <td>0.418169</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>COLE_Tec_ac</th>
      <td>1161.0</td>
      <td>0.262705</td>
      <td>0.440293</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>COLE_otro_car</th>
      <td>1161.0</td>
      <td>0.001723</td>
      <td>0.041487</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>COLE_completa</th>
      <td>1161.0</td>
      <td>0.187721</td>
      <td>0.386584</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>COLE_mañana</th>
      <td>1161.0</td>
      <td>0.642779</td>
      <td>0.460690</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>COLE_noche</th>
      <td>1161.0</td>
      <td>0.014181</td>
      <td>0.079981</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>COLE_sabatina</th>
      <td>1161.0</td>
      <td>0.035823</td>
      <td>0.139835</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>COLE_tarde</th>
      <td>1161.0</td>
      <td>0.029246</td>
      <td>0.157346</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>COLE_unica</th>
      <td>1161.0</td>
      <td>0.090250</td>
      <td>0.280816</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>nse_1</th>
      <td>1161.0</td>
      <td>0.397071</td>
      <td>0.489502</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>nse_2</th>
      <td>1161.0</td>
      <td>0.586563</td>
      <td>0.492662</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>nse_3</th>
      <td>1161.0</td>
      <td>0.016365</td>
      <td>0.126930</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>nse_4</th>
      <td>1161.0</td>
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




```python
x = df_f.estudiantes
plt.hist(x, bins= 5, color= 'green')
plt.title('Histograma estudiantes por sede educativa')
plt.show()
```


    
![png](output_15_0.png)
    


Podemos observar que la mayoria de las sede educativas se concentran a la izquierda, esto quiere decir que la mayoria son sedes pequeñas que tienen entre 1 a 50 estudiantes en los cursos de 11 para 2019 - 2. 

Por esta razon se propone analizar sedes educativas con un número de estudiantes comparable.


```python
sedes_nounicas = df_f.estudiantes > 2 
df1 = df_f[sedes_nounicas]
print(df1.shape)
df1
#Esto se hace para que las sedes educativas analizadas no dependan solo de un estudiante 
```

    (1129, 45)





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
      <th>5</th>
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
      <th>6</th>
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
      <th>2322</th>
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
      <th>2323</th>
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
      <th>2325</th>
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
      <th>2342</th>
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
      <th>2343</th>
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
<p>1129 rows × 45 columns</p>
</div>




```python
sedes_pequeñas = df1.estudiantes < 50 
df = df1[sedes_pequeñas]
print(df.shape)
df
#Bajo este filtra de analisis para las sedes educativas pequeñas terminamos con 989 observaciones 
```

    (989, 45)





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
      <th>5</th>
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
      <th>6</th>
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
      <th>2322</th>
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
      <th>2323</th>
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
      <th>2325</th>
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
      <th>2342</th>
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
      <th>2343</th>
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
x = df.estudiantes
plt.hist(x, bins= 15, color= 'purple')
plt.title('Histograma estudiantes por sede educativa pequeña')
plt.show()

#Observamos que la distribución de los estudiantes sigue consentrada hacia la derecha
```


    
![png](output_19_0.png)
    


De este modo, aunque aun tenemos una cola larga hacia la derecha, tenemos sedes educativas mucho más comparables por su tamaño. Las observaciones ahora son de 989 sedes educativas.


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



El 31.7% de las sedes educativas analizadas tiene un Kiosco Vive Digital dentro de sus instalaciones. Por otro lado, el porcentaje de estudiantes niñas es un poco más alto que el de los niños casi por 7pps. Al contrario, el porcentaje de estudiantes de comunidades étnicas es de solo 16,7%, la mayoría de los estudiantes (46,9%) leen entre 30 minutos o menos, mientras que navegan (33,9%) entre 30 y 60 minutos. Todas las sede educativas son de calendario A, el 99% son oficiales, el 97% no es bilingüe, más de 50% son de carácter académico con jornada por la mañana y de nivel socioeconómico muy bajo o bajo.


```python
figure = plt.figure(figsize=(9,5), dpi=100)    
graph = figure.add_subplot(111)
freq = pd.value_counts(df.Si_KVD)
plt.title('Sedes educativas con Kiosco Vive Digital')
bins = freq.index
x=graph.bar(bins,freq.values) #deja por fuera los Nan
plt.xticks(rotation=90);
```


    
![png](output_23_0.png)
    



```python
count = df['Si_KVD'].value_counts()
count
```




    0    675
    1    314
    Name: Si_KVD, dtype: int64




```python
x = df.PUNT_GLOBAL
plt.hist(x, bins= 20, color= 'darkblue')
plt.title('Histograma puntaje global medio por sede educativa pequeña')
plt.show()
```


    
![png](output_25_0.png)
    



```python
x = df.PUNT_MATEMATICAS
plt.hist(x, bins= 20, color= 'darkgreen')
plt.title('Histograma puntage en matemáticas medio por sede educativa pequeña')
plt.show()
```


    
![png](output_26_0.png)
    



```python
x = df.PUNT_INGLES
plt.hist(x, bins= 20, color= 'orange')
plt.title('Histograma puntaje en ingles medio por sede educativa pequeña')
plt.show()
```


    
![png](output_27_0.png)
    


En general las distribuciones de los puntajes se ven normales.


```python
df.to_csv('~/Downloads/ICFES_KVD_1.csv', index = False)
```

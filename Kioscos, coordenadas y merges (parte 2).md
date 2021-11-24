# Base Kioscos Vive Digital


```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
```

Para adquirir la información de los Kioscos Vive digital, de ahora en adelande KVD, se utiliza la API de la pagina Datos Abiertos que es gestionada por el Ministerio de Tecnologías de la Información y las Comunicaciones


```python
#Libreria con la que trabaja el API de Datos Abiertos 
from sodapy import Socrata
```


```python
#La base de KVD es de uso público por lo que no se requiere de una cuenta o token
KVD = Socrata ("www.datos.gov.co", None)
Kioscos = KVD.get_metadata("prt6-8t9a")
#En limit se utiliza el numero total de observaciones de la base de datos 
df = KVD.get("prt6-8t9a", content_type = "csv", limit=6816)
```

    WARNING:root:Requests made without an app_token will be subject to strict throttling limits.



```python
#Dar fomato a la base de datos 
columns = df[0]
del(df[0])
df = pd.DataFrame(df)
df.columns = columns
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
      <th>operador</th>
      <th>codigo_dircon</th>
      <th>dane_departamento</th>
      <th>departamento</th>
      <th>dane_municipio</th>
      <th>municipio</th>
      <th>dane_centro_poblado</th>
      <th>centro_poblado</th>
      <th>direccion_del_kioscos</th>
      <th>dane_institucion_educativa</th>
      <th>...</th>
      <th>coordenadas_de_referencia_1</th>
      <th>horario_entre_semana</th>
      <th>horario_fin_de_semana</th>
      <th>fecha_inauguracion</th>
      <th>inaugurado_por</th>
      <th>fase</th>
      <th>valor_kvd</th>
      <th>fechacorte</th>
      <th>idindicadoraspa</th>
      <th>vigencia</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Uni�n temporal KVD</td>
      <td>54232</td>
      <td>99</td>
      <td>VICHADA</td>
      <td>99773</td>
      <td>CUMARIBO</td>
      <td>99773000</td>
      <td>ASOCORTOMO</td>
      <td>ASOCORTOMO</td>
      <td>299773000064</td>
      <td>...</td>
      <td>-70,33694444</td>
      <td>08:00 - 12:00</td>
      <td></td>
      <td>2015-11-12</td>
      <td>OPERADOR</td>
      <td>FASE 2</td>
      <td>164470428,52</td>
      <td>2015-12-31</td>
      <td>0</td>
      <td>2015-12-31</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Uni�n Temporal Kioscos 2</td>
      <td>50690</td>
      <td>76</td>
      <td>VALLE DEL CAUCA</td>
      <td>76616</td>
      <td>RIOFRIO</td>
      <td>76616002</td>
      <td>FENICIA</td>
      <td>CORREG FENICIA</td>
      <td>276616000590</td>
      <td>...</td>
      <td>-76,38566667</td>
      <td>2 pm - 6 pm</td>
      <td>8 am - 5 pm</td>
      <td>2015-08-20</td>
      <td>OPERADOR</td>
      <td>FASE 2</td>
      <td>160896602,48</td>
      <td>2015-12-31</td>
      <td>0</td>
      <td>2015-12-31</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Uni�n temporal KVD</td>
      <td>54320</td>
      <td>99</td>
      <td>VICHADA</td>
      <td>99773</td>
      <td>CUMARIBO</td>
      <td>99773000</td>
      <td>SAN LUIS DE ZAMA</td>
      <td>SAN LUIS DE ZAMA</td>
      <td>299572000074</td>
      <td>...</td>
      <td>-67,85916667</td>
      <td>08:00 - 15:00</td>
      <td></td>
      <td>1900-01-01</td>
      <td>ND</td>
      <td>FASE 2</td>
      <td>164470428,52</td>
      <td>2015-12-31</td>
      <td>0</td>
      <td>2015-12-31</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Uni�n Temporal Kioscos 2</td>
      <td>50606</td>
      <td>76</td>
      <td>VALLE DEL CAUCA</td>
      <td>76520</td>
      <td>PALMIRA</td>
      <td>76520001</td>
      <td>AGUAS CLARAS</td>
      <td>VEREDA AGUACLARA</td>
      <td>276520002320</td>
      <td>...</td>
      <td>-76,23583333</td>
      <td>2 pm - 6 pm</td>
      <td>8 am - 5 pm</td>
      <td>2015-05-08</td>
      <td>OPERADOR</td>
      <td>FASE 2</td>
      <td>160896602,48</td>
      <td>2015-12-31</td>
      <td>0</td>
      <td>2015-12-31</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Uni�n Temporal Kioscos 2</td>
      <td>50413</td>
      <td>76</td>
      <td>VALLE DEL CAUCA</td>
      <td>76318</td>
      <td>GUACARI</td>
      <td>76318003</td>
      <td>CORREGIMIENTO GUABAS</td>
      <td>CORREGIMIENTO GUABAS</td>
      <td>276318000111</td>
      <td>...</td>
      <td>-76,21000000</td>
      <td>13:00 - 17:00</td>
      <td></td>
      <td>2015-02-24</td>
      <td>OPERADOR</td>
      <td>FASE 2</td>
      <td>160896602,48</td>
      <td>2015-12-31</td>
      <td>0</td>
      <td>2015-12-31</td>
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
      <th>6811</th>
      <td>Uni�n Temporal Kioscos 2</td>
      <td>39211</td>
      <td>50</td>
      <td>META</td>
      <td>50573</td>
      <td>PUERTO LOPEZ</td>
      <td>50573010</td>
      <td>INSPECCI�N. BOCAS DEL GUAYURIBA</td>
      <td>INSPECCI�N. BOCAS DEL GUAYURIBA</td>
      <td>250573000231</td>
      <td>...</td>
      <td>-73,09166667</td>
      <td>01:00 PM - 05:00 PM</td>
      <td>N/A</td>
      <td>2015-12-11</td>
      <td>OPERADOR</td>
      <td>FASE 2</td>
      <td>160896602,48</td>
      <td>2014-12-31</td>
      <td>0</td>
      <td>2014-12-31</td>
    </tr>
    <tr>
      <th>6812</th>
      <td>SKYNET DE COLOMBIA</td>
      <td>28795</td>
      <td>19</td>
      <td>CAUCA</td>
      <td>19355</td>
      <td>INZA</td>
      <td>19355000</td>
      <td>VEREDA COSCURO</td>
      <td>VEREDA COSCURO</td>
      <td>219355000283</td>
      <td>...</td>
      <td>-76,10009389</td>
      <td>14:00 - 18:00</td>
      <td></td>
      <td>1900-01-01</td>
      <td>ND</td>
      <td>FASE 3</td>
      <td>118120953,67</td>
      <td>2016-12-31</td>
      <td>0</td>
      <td>2016-12-31</td>
    </tr>
    <tr>
      <th>6813</th>
      <td>UT BT-INRED K3</td>
      <td>22440</td>
      <td>05</td>
      <td>ANTIOQUIA</td>
      <td>05895</td>
      <td>ZARAGOZA</td>
      <td>5895000</td>
      <td>LA VALENTINA</td>
      <td>VEREDA LA VALENTINA</td>
      <td>205895000320</td>
      <td>...</td>
      <td>-74,93083333</td>
      <td>07:00 - 01:30</td>
      <td>07:00 - 01:30</td>
      <td>1900-01-01</td>
      <td>ND</td>
      <td>FASE 3</td>
      <td>139247670,07</td>
      <td>2016-12-31</td>
      <td>0</td>
      <td>2016-12-31</td>
    </tr>
    <tr>
      <th>6814</th>
      <td>Uni�n Temporal Internet para Kioscos</td>
      <td>30088</td>
      <td>20</td>
      <td>CESAR</td>
      <td>20001</td>
      <td>VALLEDUPAR</td>
      <td>20001031</td>
      <td>RIO SECO</td>
      <td>R�O SECO</td>
      <td>220001066820</td>
      <td>...</td>
      <td>-73,33444400</td>
      <td>14:00 - 17:30</td>
      <td>08:30 - 17:00</td>
      <td>2015-02-13</td>
      <td>GESTOR, RECTOR, COMUNIDAD</td>
      <td>FASE 2</td>
      <td>138323795,51</td>
      <td>2014-12-31</td>
      <td>0</td>
      <td>2014-12-31</td>
    </tr>
    <tr>
      <th>6815</th>
      <td>Uni�n Temporal Kioscos 2</td>
      <td>29898</td>
      <td>19</td>
      <td>CAUCA</td>
      <td>19807</td>
      <td>TIMBIO</td>
      <td>19807000</td>
      <td>VEREDA TUNURCO</td>
      <td>VEREDA TUNURCO</td>
      <td>219807000251</td>
      <td>...</td>
      <td>-76,69100000</td>
      <td>7:00 A 13:00</td>
      <td>NA</td>
      <td>2014-12-01</td>
      <td>OPERADOR</td>
      <td>FASE 2</td>
      <td>160896602,48</td>
      <td>2015-12-31</td>
      <td>0</td>
      <td>2015-12-31</td>
    </tr>
  </tbody>
</table>
<p>6816 rows × 46 columns</p>
</div>




```python
#Revisar qué columnas son de interes 
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
      <td>operador</td>
    </tr>
    <tr>
      <th>1</th>
      <td>codigo_dircon</td>
    </tr>
    <tr>
      <th>2</th>
      <td>dane_departamento</td>
    </tr>
    <tr>
      <th>3</th>
      <td>departamento</td>
    </tr>
    <tr>
      <th>4</th>
      <td>dane_municipio</td>
    </tr>
    <tr>
      <th>5</th>
      <td>municipio</td>
    </tr>
    <tr>
      <th>6</th>
      <td>dane_centro_poblado</td>
    </tr>
    <tr>
      <th>7</th>
      <td>centro_poblado</td>
    </tr>
    <tr>
      <th>8</th>
      <td>direccion_del_kioscos</td>
    </tr>
    <tr>
      <th>9</th>
      <td>dane_institucion_educativa</td>
    </tr>
    <tr>
      <th>10</th>
      <td>institucion_educativa</td>
    </tr>
    <tr>
      <th>11</th>
      <td>dane_sede</td>
    </tr>
    <tr>
      <th>12</th>
      <td>nombre_sede</td>
    </tr>
    <tr>
      <th>13</th>
      <td>numero_telefonico_del_kiosco</td>
    </tr>
    <tr>
      <th>14</th>
      <td>numero_telefonico_del_kiosco_1</td>
    </tr>
    <tr>
      <th>15</th>
      <td>tipo_de_conectividad</td>
    </tr>
    <tr>
      <th>16</th>
      <td>tipo_kvd</td>
    </tr>
    <tr>
      <th>17</th>
      <td>tipo_de_energia</td>
    </tr>
    <tr>
      <th>18</th>
      <td>no_de_contratooperador</td>
    </tr>
    <tr>
      <th>19</th>
      <td>ano_de_contratooperador</td>
    </tr>
    <tr>
      <th>20</th>
      <td>no_de_contratointerventoria</td>
    </tr>
    <tr>
      <th>21</th>
      <td>ano_de_contratointerventoria</td>
    </tr>
    <tr>
      <th>22</th>
      <td>nombre_del_coordinador</td>
    </tr>
    <tr>
      <th>23</th>
      <td>email_coordinador</td>
    </tr>
    <tr>
      <th>24</th>
      <td>telefono_coordinador</td>
    </tr>
    <tr>
      <th>25</th>
      <td>nombre_gestor</td>
    </tr>
    <tr>
      <th>26</th>
      <td>email_gestor</td>
    </tr>
    <tr>
      <th>27</th>
      <td>telefono_gestor</td>
    </tr>
    <tr>
      <th>28</th>
      <td>estado_del_kiosco</td>
    </tr>
    <tr>
      <th>29</th>
      <td>meta</td>
    </tr>
    <tr>
      <th>30</th>
      <td>inicio_de_operacion</td>
    </tr>
    <tr>
      <th>31</th>
      <td>fin_de_operacion</td>
    </tr>
    <tr>
      <th>32</th>
      <td>velocidadsubida_kb</td>
    </tr>
    <tr>
      <th>33</th>
      <td>velocidadbajada_kb</td>
    </tr>
    <tr>
      <th>34</th>
      <td>dda</td>
    </tr>
    <tr>
      <th>35</th>
      <td>coordenadas_de_referencia</td>
    </tr>
    <tr>
      <th>36</th>
      <td>coordenadas_de_referencia_1</td>
    </tr>
    <tr>
      <th>37</th>
      <td>horario_entre_semana</td>
    </tr>
    <tr>
      <th>38</th>
      <td>horario_fin_de_semana</td>
    </tr>
    <tr>
      <th>39</th>
      <td>fecha_inauguracion</td>
    </tr>
    <tr>
      <th>40</th>
      <td>inaugurado_por</td>
    </tr>
    <tr>
      <th>41</th>
      <td>fase</td>
    </tr>
    <tr>
      <th>42</th>
      <td>valor_kvd</td>
    </tr>
    <tr>
      <th>43</th>
      <td>fechacorte</td>
    </tr>
    <tr>
      <th>44</th>
      <td>idindicadoraspa</td>
    </tr>
    <tr>
      <th>45</th>
      <td>vigencia</td>
    </tr>
  </tbody>
</table>
</div>




```python
#Columnas de interes y creación de otro dataframe más pequeño
df2 = df.iloc[:,[4,9,11,12,16,28]]
df2.shape
```




    (6816, 6)




```python
#Dejar solo los KVD que se encuentran en sedes educativas y que estan en operación
educativa = df2.tipo_kvd == 'SEDE EDUCATIVA'
df2_K = df2[educativa]
activa = df2.estado_del_kiosco == 'EN OPERACION'
df2_KVD = df2_K[activa]
print(df2_KVD.shape)
df2_KVD

#De 6.816 KVD a 6.552 que estan activos y en sedes educativas
```

    (6552, 6)


    <ipython-input-7-f22ff1e9ca28>:5: UserWarning: Boolean Series key will be reindexed to match DataFrame index.
      df2_KVD = df2_K[activa]





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
      <th>dane_municipio</th>
      <th>dane_institucion_educativa</th>
      <th>dane_sede</th>
      <th>nombre_sede</th>
      <th>tipo_kvd</th>
      <th>estado_del_kiosco</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>99773</td>
      <td>299773000064</td>
      <td>299773000064</td>
      <td>C.E. INTERNADO ASOCORTOMO - SEDE PRINCIPAL</td>
      <td>SEDE EDUCATIVA</td>
      <td>EN OPERACION</td>
    </tr>
    <tr>
      <th>1</th>
      <td>76616</td>
      <td>276616000590</td>
      <td>276616000204</td>
      <td>LAS AMERICAS</td>
      <td>SEDE EDUCATIVA</td>
      <td>EN OPERACION</td>
    </tr>
    <tr>
      <th>2</th>
      <td>99773</td>
      <td>299572000074</td>
      <td>299773002989</td>
      <td>SAN LUIS DE ZAMA</td>
      <td>SEDE EDUCATIVA</td>
      <td>EN OPERACION</td>
    </tr>
    <tr>
      <th>3</th>
      <td>76520</td>
      <td>276520002320</td>
      <td>276520002117</td>
      <td>SAN JUAN BAUTISTA</td>
      <td>SEDE EDUCATIVA</td>
      <td>EN OPERACION</td>
    </tr>
    <tr>
      <th>4</th>
      <td>76318</td>
      <td>276318000111</td>
      <td>276318000111</td>
      <td>IE JOSE CELESTINO MUTIS</td>
      <td>SEDE EDUCATIVA</td>
      <td>EN OPERACION</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>6811</th>
      <td>50573</td>
      <td>250573000231</td>
      <td>250573000036</td>
      <td>BOCAS DEL GUAYURIBA</td>
      <td>SEDE EDUCATIVA</td>
      <td>EN OPERACION</td>
    </tr>
    <tr>
      <th>6812</th>
      <td>19355</td>
      <td>219355000283</td>
      <td>219355000321</td>
      <td>ESCUELA RURAL MIXTA COSCURO</td>
      <td>SEDE EDUCATIVA</td>
      <td>EN OPERACION</td>
    </tr>
    <tr>
      <th>6813</th>
      <td>05895</td>
      <td>205895000320</td>
      <td>205895000842</td>
      <td>CHILONA CENTRAL</td>
      <td>SEDE EDUCATIVA</td>
      <td>EN OPERACION</td>
    </tr>
    <tr>
      <th>6814</th>
      <td>20001</td>
      <td>220001066820</td>
      <td>220001002031</td>
      <td>SAN FERNANDO DE RIO SECO</td>
      <td>SEDE EDUCATIVA</td>
      <td>EN OPERACION</td>
    </tr>
    <tr>
      <th>6815</th>
      <td>19807</td>
      <td>219807000251</td>
      <td>219807000367</td>
      <td>TUNURCO</td>
      <td>SEDE EDUCATIVA</td>
      <td>EN OPERACION</td>
    </tr>
  </tbody>
</table>
<p>6552 rows × 6 columns</p>
</div>




```python
#Revisamos si hay duplicados 
df2_KVD.duplicated(subset=['dane_sede']).sum()
```




    4




```python
#Ver cuáles son los duplicados 
duplicate_KVD = df2_KVD.duplicated(subset=['dane_sede'])
if duplicate_KVD.any():
   print(df2_KVD.loc[duplicate_KVD], end='\n\n')
```

         dane_municipio dane_institucion_educativa     dane_sede  \
    1498          13654                          0             0   
    1765          47980               247980002420  247980002420   
    3512          27430               227077000369  227077001748   
    5057          19809                          0             0   
    
                                    nombre_sede        tipo_kvd estado_del_kiosco  
    1498  ORGANIZACI�N PRO NI�EZ INDEFENSA-OPNI  SEDE EDUCATIVA      EN OPERACION  
    1765                CON DE DESARROLLO RURAL  SEDE EDUCATIVA      EN OPERACION  
    3512           SIMON BOLIVAR DE PABLO SEXTO  SEDE EDUCATIVA      EN OPERACION  
    5057  ORGANIZACI�N PRO NI�EZ INDEFENSA-OPNI  SEDE EDUCATIVA      EN OPERACION  
    



```python
#Eliminar los duplicados 
df2_KVD.drop_duplicates(subset=['dane_sede'], inplace=True)
df2_KVD.duplicated(subset=['dane_sede']).sum()
```

    <ipython-input-10-9902437e062f>:2: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      df2_KVD.drop_duplicates(subset=['dane_sede'], inplace=True)





    0



# Merge - 1

Como llave puede ser: 
- df2_KVD["dane_sede"]
- df_ICFES["COLE_COD_DANE_SEDE"]


```python
#Se utiliza la base de la parte uno 
df_ICFES = pd.read_csv("ICFES_Lina.csv")
print(df_ICFES.shape)
df_ICFES
```

    (2344, 43)





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
      <td>1,05001E+11</td>
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
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
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
      <td>1,05001E+11</td>
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
      <td>0.0</td>
      <td>1.0</td>
      <td>0.000000</td>
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
      <td>1,05148E+11</td>
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
      <td>1.0</td>
      <td>0.0</td>
      <td>0.000000</td>
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
      <td>1,0544E+11</td>
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
      <td>1.0</td>
      <td>0.0</td>
      <td>0.000000</td>
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
      <td>1,05679E+11</td>
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
      <td>0.0</td>
      <td>0.0</td>
      <td>0.252174</td>
      <td>0.0</td>
      <td>0.747826</td>
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
      <th>2339</th>
      <td>4,85001E+11</td>
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
      <td>1.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2340</th>
      <td>4,86001E+11</td>
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
      <td>1.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2341</th>
      <td>4,86865E+11</td>
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
      <td>0.0</td>
      <td>1.0</td>
      <td>0.000000</td>
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
      <td>4,99001E+11</td>
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
      <td>1.0</td>
      <td>0.0</td>
      <td>0.000000</td>
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
      <td>4,99001E+11</td>
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
      <td>1.0</td>
      <td>0.0</td>
      <td>0.000000</td>
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
<p>2344 rows × 43 columns</p>
</div>




```python
a = pd.merge(df_ICFES, df2_KVD, right_on = ["dane_sede"],left_on = ["COLE_COD_DANE_SEDE"],how = 'left')
```


```python
print(a.shape)
a
```

    (2344, 49)





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
      <th>nse_1</th>
      <th>nse_2</th>
      <th>nse_3</th>
      <th>nse_4</th>
      <th>dane_municipio</th>
      <th>dane_institucion_educativa</th>
      <th>dane_sede</th>
      <th>nombre_sede</th>
      <th>tipo_kvd</th>
      <th>estado_del_kiosco</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1,05001E+11</td>
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
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1,05001E+11</td>
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
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1,05148E+11</td>
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
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1,0544E+11</td>
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
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1,05679E+11</td>
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
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
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
      <td>4,85001E+11</td>
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
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2340</th>
      <td>4,86001E+11</td>
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
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2341</th>
      <td>4,86865E+11</td>
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
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2342</th>
      <td>4,99001E+11</td>
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
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2343</th>
      <td>4,99001E+11</td>
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
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
<p>2344 rows × 49 columns</p>
</div>




```python
a.isna().sum() 
```




    COLE_COD_DANE_SEDE               0
    COLE_NOMBRE_SEDE                 0
    COLE_COD_MCPIO_UBICACION         0
    estudiantes                      0
    PUNT_GLOBAL                      0
    PUNT_LECTURA_CRITICA             0
    PUNT_MATEMATICAS                 0
    PUNT_INGLES                      0
    F                                0
    M                                0
    Si_etnia                         0
    NO_etnia                         0
    lec_1                            0
    lec_2                            0
    lec_3                            0
    lec_4                            0
    lec_5                            0
    int_1                            0
    int_2                            0
    int_3                            0
    int_4                            0
    int_5                            0
    No_oficial                       0
    Oficial                          0
    A_Calendario                     0
    B_Calendario                     0
    Otro_Calendario                  0
    No_BILINGUE                      0
    Si_BILINGUE                      0
    COLE_Academic                    0
    COLE_Tecnic                      0
    COLE_Tec_ac                      0
    COLE_otro_car                    0
    COLE_completa                    0
    COLE_mañana                      0
    COLE_noche                       0
    COLE_sabatina                    0
    COLE_tarde                       0
    COLE_unica                       0
    nse_1                            0
    nse_2                            0
    nse_3                            0
    nse_4                            0
    dane_municipio                2344
    dane_institucion_educativa    2344
    dane_sede                     2344
    nombre_sede                   2344
    tipo_kvd                      2344
    estado_del_kiosco             2344
    dtype: int64



Sin embargo, todas las observaciones tienes missing en las columnas de la base de KVD.

Al observar las columnas mencionadas, los codigos de la institución educativa empiezan en 2. En la base del ICFES tambien hay codigos de establecimiento que empiezan con 2 pero se leen como string puesto que los codigos estan en notación cientifica con E+11 que, haciendo la conversión, haria que todos los codigos termine en cero (0). Estos codigos siguen siendo diferentes a los encontrados en la base de KVD. 


```python
#Intento de cambiar el formato de los codigos en la base de datos quitando el "string" de la notación cientifica
df_ICFES['COLE_COD_DANE_SEDE'] = df_ICFES['COLE_COD_DANE_SEDE'].map(lambda x: str(x)[:-4])
df_ICFES
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
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
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
      <td>0.0</td>
      <td>1.0</td>
      <td>0.000000</td>
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
      <td>1.0</td>
      <td>0.0</td>
      <td>0.000000</td>
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
      <td>1.0</td>
      <td>0.0</td>
      <td>0.000000</td>
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
      <td>0.0</td>
      <td>0.0</td>
      <td>0.252174</td>
      <td>0.0</td>
      <td>0.747826</td>
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
      <td>1.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
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
      <td>1.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
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
      <td>0.0</td>
      <td>1.0</td>
      <td>0.000000</td>
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
      <td>1.0</td>
      <td>0.0</td>
      <td>0.000000</td>
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
      <td>1.0</td>
      <td>0.0</td>
      <td>0.000000</td>
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
<p>2344 rows × 43 columns</p>
</div>



En la base del ICFES aparecen sedes con el mismo codigo de sede  pero con nombres diferentes. Esto puede indicar que ninguno de estos codigos por si solos puede servir como identificador, ni tampoco como llave para hacer el merge. 

Teniendo en cuenta que el unico identificar en la base del ICFES es el nombre de la sede se comprueba que los nombres sean iguales para hacer el merge por estas variables


```python
#Verificar la existencia de duplicados 
df_ICFES.duplicated(subset=['COLE_NOMBRE_SEDE']).sum()
```




    60



Ya que existen duplicados en los nombres de las sedes, se realiza el merge a partir del nombre de la sede educativa y el codigo del municipio en el que esta ubicado.  COLE_COD_MCPIO_UBICACION


```python
#Prueba con las variables mencionadas 
df_ICFES.duplicated(subset=['COLE_NOMBRE_SEDE','COLE_COD_MCPIO_UBICACION']).sum()
```




    0



## Merge - 2
- df2_KVD["nombre_sede","dane_municipio"]
- df_ICFES["COLE_NOMBRE_SEDE", "COLE_COD_MCPIO_UBICACION"]


```python
#Se cambia el formato de la columna dane_municipio ya que es un objeto mientras que en la base de ICFES es un entero
df2_KVD['dane_municipio'] = pd.to_numeric(df2_KVD['dane_municipio'])
```

    <ipython-input-18-3ee64691c347>:2: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      df2_KVD['dane_municipio'] = pd.to_numeric(df2_KVD['dane_municipio'])



```python
df2_KVD.dtypes
```




    dane_municipio                 int64
    dane_institucion_educativa    object
    dane_sede                     object
    nombre_sede                   object
    tipo_kvd                      object
    estado_del_kiosco             object
    dtype: object




```python
# Se realiza el merge teniendo dos llaves para la identificación de observaciones unicas
a = pd.merge(df_ICFES, df2_KVD, right_on = ["nombre_sede","dane_municipio"],
   left_on = ["COLE_NOMBRE_SEDE", "COLE_COD_MCPIO_UBICACION"],
   how = 'left')
pd.options.display.max_columns = None
print(a.shape)
a
#La base de datos tiene 2 observaciones de más por lo que se asume que existen duplicados 
```

    (2346, 49)





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
      <th>Si_etnia</th>
      <th>NO_etnia</th>
      <th>lec_1</th>
      <th>lec_2</th>
      <th>lec_3</th>
      <th>lec_4</th>
      <th>lec_5</th>
      <th>int_1</th>
      <th>int_2</th>
      <th>int_3</th>
      <th>int_4</th>
      <th>int_5</th>
      <th>No_oficial</th>
      <th>Oficial</th>
      <th>A_Calendario</th>
      <th>B_Calendario</th>
      <th>Otro_Calendario</th>
      <th>No_BILINGUE</th>
      <th>Si_BILINGUE</th>
      <th>COLE_Academic</th>
      <th>COLE_Tecnic</th>
      <th>COLE_Tec_ac</th>
      <th>COLE_otro_car</th>
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
      <th>dane_municipio</th>
      <th>dane_institucion_educativa</th>
      <th>dane_sede</th>
      <th>nombre_sede</th>
      <th>tipo_kvd</th>
      <th>estado_del_kiosco</th>
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
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.212766</td>
      <td>0.319149</td>
      <td>0.255319</td>
      <td>0.191489</td>
      <td>0.021277</td>
      <td>0.000000</td>
      <td>0.255319</td>
      <td>0.319149</td>
      <td>0.297872</td>
      <td>0.127660</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>1.000000</td>
      <td>0.0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
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
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.130435</td>
      <td>0.521739</td>
      <td>0.130435</td>
      <td>0.086957</td>
      <td>0.130435</td>
      <td>0.043478</td>
      <td>0.130435</td>
      <td>0.347826</td>
      <td>0.304348</td>
      <td>0.173913</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
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
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.666667</td>
      <td>0.333333</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.166667</td>
      <td>0.250000</td>
      <td>0.083333</td>
      <td>0.416667</td>
      <td>0.083333</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1</td>
      <td>0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
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
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.173913</td>
      <td>0.434783</td>
      <td>0.347826</td>
      <td>0.043478</td>
      <td>0.000000</td>
      <td>0.086957</td>
      <td>0.043478</td>
      <td>0.391304</td>
      <td>0.434783</td>
      <td>0.043478</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0</td>
      <td>0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
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
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.173913</td>
      <td>0.460870</td>
      <td>0.286957</td>
      <td>0.060870</td>
      <td>0.017391</td>
      <td>0.069565</td>
      <td>0.147826</td>
      <td>0.330435</td>
      <td>0.295652</td>
      <td>0.156522</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.252174</td>
      <td>0.0</td>
      <td>0.747826</td>
      <td>0.0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
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
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>2341</th>
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
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.075472</td>
      <td>0.301887</td>
      <td>0.320755</td>
      <td>0.283019</td>
      <td>0.018868</td>
      <td>0.000000</td>
      <td>0.094340</td>
      <td>0.226415</td>
      <td>0.471698</td>
      <td>0.207547</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2342</th>
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
      <td>0.250000</td>
      <td>0.750000</td>
      <td>0.000000</td>
      <td>0.500000</td>
      <td>0.250000</td>
      <td>0.250000</td>
      <td>0.000000</td>
      <td>0.333333</td>
      <td>0.333333</td>
      <td>0.333333</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2343</th>
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
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.375000</td>
      <td>0.375000</td>
      <td>0.250000</td>
      <td>0.000000</td>
      <td>0.500000</td>
      <td>0.125000</td>
      <td>0.375000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>86757.0</td>
      <td>486865000961</td>
      <td>486865000961</td>
      <td>I.E.R. SAN MARCELINO - SEDE PRINCIPAL</td>
      <td>SEDE EDUCATIVA</td>
      <td>EN OPERACION</td>
    </tr>
    <tr>
      <th>2344</th>
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
      <td>0.710526</td>
      <td>0.289474</td>
      <td>0.131579</td>
      <td>0.342105</td>
      <td>0.368421</td>
      <td>0.105263</td>
      <td>0.052632</td>
      <td>0.289474</td>
      <td>0.105263</td>
      <td>0.184211</td>
      <td>0.315789</td>
      <td>0.105263</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0</td>
      <td>0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>99773.0</td>
      <td>499001001919</td>
      <td>499001001919</td>
      <td>I.E. INTERNADO SANTA TERESITA DEL TUPARRO - SE...</td>
      <td>SEDE EDUCATIVA</td>
      <td>EN OPERACION</td>
    </tr>
    <tr>
      <th>2345</th>
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
      <td>0.857143</td>
      <td>0.142857</td>
      <td>0.071429</td>
      <td>0.428571</td>
      <td>0.214286</td>
      <td>0.142857</td>
      <td>0.142857</td>
      <td>0.285714</td>
      <td>0.142857</td>
      <td>0.357143</td>
      <td>0.071429</td>
      <td>0.142857</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0</td>
      <td>0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>99524.0</td>
      <td>499001001170</td>
      <td>499001001170</td>
      <td>I.E. INTERNADO TEHODORO WEIJNEN - SEDE PRINCIPAL</td>
      <td>SEDE EDUCATIVA</td>
      <td>EN OPERACION</td>
    </tr>
  </tbody>
</table>
<p>2346 rows × 49 columns</p>
</div>



Para probar la existencia de duplicados se realiza una columna en la que se concatenan el nombre de la sede con el codigo del municipio. 


```python
b = a['COLE_COD_MCPIO_UBICACION'].apply(str)
```


```python
a['COLE_NOMBRE_SEDE-COLE_COD_MCPIO_UBICACION'] = a['COLE_NOMBRE_SEDE'] + ' - ' + b
a['COLE_NOMBRE_SEDE-COLE_COD_MCPIO_UBICACION']
```




    0                INST EDUC FE Y ALEGRIA EL LIMONAR - 5001
    1                           INST EDUC LOMA HERMOSA - 5001
    2               I. E. R. CAMPESTRE NUEVO HORIZONTE - 5148
    3                   INSTITUTO TECNICO DE MARINILLA - 5440
    4                              LICEO TOMAS EASTMAN - 5679
                                  ...                        
    2341                       GIMNASIO DE LOS LLANOS - 85001
    2342            I.E.R. MAYOYOQUE - SEDE PRINCIPAL - 86571
    2343        I.E.R. SAN MARCELINO - SEDE PRINCIPAL - 86757
    2344    I.E. INTERNADO SANTA TERESITA DEL TUPARRO - SE...
    2345    I.E. INTERNADO TEHODORO WEIJNEN - SEDE PRINCIP...
    Name: COLE_NOMBRE_SEDE-COLE_COD_MCPIO_UBICACION, Length: 2346, dtype: object




```python
a.duplicated(subset=['COLE_NOMBRE_SEDE-COLE_COD_MCPIO_UBICACION']).sum()
```




    2




```python
a.drop_duplicates(subset=['COLE_NOMBRE_SEDE-COLE_COD_MCPIO_UBICACION'], inplace=True)
```


```python
a.duplicated(subset=['COLE_NOMBRE_SEDE-COLE_COD_MCPIO_UBICACION']).sum()
```




    0




```python
a.shape 
#Se cuenta con una base de 2.344 observaciones con 50 columnas, no ha cambiado el número de obsevaciones
```




    (2344, 50)



# Merge con latitud de las sedes educativas


```python
#La base de sedes educativas es de uso público por lo que no se requiere de una cuenta o token
Coor = Socrata ("www.datos.gov.co", None)
coordenadas = Coor.get_metadata("x5ay-984n")
#En limit se utiliza el numero total de observaciones de la base de datos 
df3 = Coor.get("x5ay-984n", content_type = "csv",limit=26000)
#Dar formato a la base de datos 
columns = df3[0]
del(df3[0])
df3 = pd.DataFrame(df3)
df3.columns = columns
df3
```

    WARNING:root:Requests made without an app_token will be subject to strict throttling limits.





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
      <th>a_o</th>
      <th>departamento</th>
      <th>secretaria</th>
      <th>cod_dane_municipio</th>
      <th>municipio</th>
      <th>codigo_dane</th>
      <th>est_id</th>
      <th>nombre_establecimiento</th>
      <th>cte_id_sector</th>
      <th>cte_id_calendario</th>
      <th>sede_id</th>
      <th>codigo_dane_sede</th>
      <th>coordenada_y_sede</th>
      <th>coordenada_x_sede</th>
      <th>nombre_sede</th>
      <th>zona</th>
      <th>direccion</th>
      <th>barrio_vereda</th>
      <th>telefono</th>
      <th>fax</th>
      <th>email</th>
      <th>principal</th>
      <th>total_matricula</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2019</td>
      <td>Caquetá</td>
      <td>CAQUETA</td>
      <td>18150</td>
      <td>Cartagena del Chairá</td>
      <td>218150000578</td>
      <td>2117</td>
      <td>I. E. R. MONSERRATE</td>
      <td>OFICIAL</td>
      <td>A</td>
      <td>43451</td>
      <td>218150001809</td>
      <td>0.50</td>
      <td>-74.18</td>
      <td>BUENAVISTA</td>
      <td>RURAL</td>
      <td>VDA BUENAVISTA</td>
      <td></td>
      <td>NO TIENE</td>
      <td></td>
      <td></td>
      <td>N</td>
      <td>12</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2019</td>
      <td>Caquetá</td>
      <td>CAQUETA</td>
      <td>18150</td>
      <td>Cartagena del Chairá</td>
      <td>218150000578</td>
      <td>2117</td>
      <td>I. E. R. MONSERRATE</td>
      <td>OFICIAL</td>
      <td>A</td>
      <td>43452</td>
      <td>218150001884</td>
      <td>1.28</td>
      <td>-74.66</td>
      <td>SANTA ELENA</td>
      <td>RURAL</td>
      <td>VDA SANTA  ELENA</td>
      <td></td>
      <td>NO TIENE</td>
      <td></td>
      <td></td>
      <td>N</td>
      <td>12</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2019</td>
      <td>Caquetá</td>
      <td>CAQUETA</td>
      <td>18150</td>
      <td>Cartagena del Chairá</td>
      <td>218150000578</td>
      <td>2117</td>
      <td>I. E. R. MONSERRATE</td>
      <td>OFICIAL</td>
      <td>A</td>
      <td>43453</td>
      <td>218150002015</td>
      <td>0.41</td>
      <td>-74.29</td>
      <td>CAÑO NEGRO</td>
      <td>RURAL</td>
      <td>VDA CAÑO NEGRO</td>
      <td></td>
      <td>NO TIENE</td>
      <td></td>
      <td></td>
      <td>N</td>
      <td>4</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2019</td>
      <td>Caquetá</td>
      <td>CAQUETA</td>
      <td>18150</td>
      <td>Cartagena del Chairá</td>
      <td>218150000578</td>
      <td>2117</td>
      <td>I. E. R. MONSERRATE</td>
      <td>OFICIAL</td>
      <td>A</td>
      <td>43454</td>
      <td>218150002031</td>
      <td>0.26</td>
      <td>-74.07</td>
      <td>CAÑO SANTO DOMINGO</td>
      <td>RURAL</td>
      <td>VDA CAÑO SANTO DOMINGO</td>
      <td></td>
      <td>NO TIENE</td>
      <td></td>
      <td></td>
      <td>N</td>
      <td>8</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2019</td>
      <td>Caquetá</td>
      <td>CAQUETA</td>
      <td>18150</td>
      <td>Cartagena del Chairá</td>
      <td>218150000578</td>
      <td>2117</td>
      <td>I. E. R. MONSERRATE</td>
      <td>OFICIAL</td>
      <td>A</td>
      <td>43456</td>
      <td>218150002104</td>
      <td>0.47</td>
      <td>-74.17</td>
      <td>SIMON BOLIVAR</td>
      <td>RURAL</td>
      <td>VDA NAPOLES</td>
      <td></td>
      <td>NO TIENE</td>
      <td></td>
      <td></td>
      <td>N</td>
      <td>17</td>
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
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>25995</th>
      <td>2019</td>
      <td>Valle del Cauca</td>
      <td>CALI</td>
      <td>76001</td>
      <td>Cali</td>
      <td>176001005091</td>
      <td>15319</td>
      <td>INSTITUCION EDUCATIVA REPUBLICA DE ISRAEL</td>
      <td>OFICIAL</td>
      <td>A</td>
      <td>72629</td>
      <td>176001005091</td>
      <td>3.46</td>
      <td>-76.51</td>
      <td>INSTITUCION EDUCATIVA REPUBLICA DE ISRAEL</td>
      <td>URBANA</td>
      <td>KR 3 43 49</td>
      <td>LAS DELICIAS</td>
      <td>4422485 / 313 5522564</td>
      <td></td>
      <td>administrador@ierdeisraelcali.edu.co</td>
      <td>S</td>
      <td>562</td>
    </tr>
    <tr>
      <th>25996</th>
      <td>2019</td>
      <td>Valle del Cauca</td>
      <td>CALI</td>
      <td>76001</td>
      <td>Cali</td>
      <td>176001040079</td>
      <td>15326</td>
      <td>INSTITUCION EDUCATIVA SANTO TOMAS</td>
      <td>OFICIAL</td>
      <td>A</td>
      <td>72630</td>
      <td>176001006453</td>
      <td>3.46</td>
      <td>-76.52</td>
      <td>JORGE ISAACS</td>
      <td>URBANA</td>
      <td>CL 30 5 88</td>
      <td>EL PORVENIR</td>
      <td>304 5672463</td>
      <td></td>
      <td>rasantotomas@gmail.com</td>
      <td>N</td>
      <td>163</td>
    </tr>
    <tr>
      <th>25997</th>
      <td>2019</td>
      <td>Valle del Cauca</td>
      <td>CALI</td>
      <td>76001</td>
      <td>Cali</td>
      <td>176001040079</td>
      <td>15326</td>
      <td>INSTITUCION EDUCATIVA SANTO TOMAS</td>
      <td>OFICIAL</td>
      <td>A</td>
      <td>72631</td>
      <td>176001003322</td>
      <td>3.47</td>
      <td>-76.52</td>
      <td>MANUELA BELTRAN</td>
      <td>URBANA</td>
      <td>KR 2 A 34 23</td>
      <td>SANTANDER</td>
      <td>304 5672501 / 304 5672463</td>
      <td></td>
      <td>rasantotomas@gmail.com</td>
      <td>N</td>
      <td>429</td>
    </tr>
    <tr>
      <th>25998</th>
      <td>2019</td>
      <td>Valle del Cauca</td>
      <td>CALI</td>
      <td>76001</td>
      <td>Cali</td>
      <td>176001040079</td>
      <td>15326</td>
      <td>INSTITUCION EDUCATIVA SANTO TOMAS</td>
      <td>OFICIAL</td>
      <td>A</td>
      <td>72633</td>
      <td>176001004132</td>
      <td>3.47</td>
      <td>-76.52</td>
      <td>SANTO TOMAS DE AQUINO</td>
      <td>URBANA</td>
      <td>CL 32 2 11</td>
      <td>BERLIN</td>
      <td>3087732 / 304 4887782</td>
      <td></td>
      <td>rasantotomas@gmail.com</td>
      <td>N</td>
      <td>318</td>
    </tr>
    <tr>
      <th>25999</th>
      <td>2019</td>
      <td>Valle del Cauca</td>
      <td>CALI</td>
      <td>76001</td>
      <td>Cali</td>
      <td>176001040079</td>
      <td>15326</td>
      <td>INSTITUCION EDUCATIVA SANTO TOMAS</td>
      <td>OFICIAL</td>
      <td>A</td>
      <td>72634</td>
      <td>176001040079</td>
      <td>3.47</td>
      <td>-76.52</td>
      <td>INSTITUCION EDUCATIVA SANTO TOMAS  CASD</td>
      <td>URBANA</td>
      <td>IND CL 34 NO 3N 15</td>
      <td>BERLIN</td>
      <td>304 5672501 / 304 5672463</td>
      <td></td>
      <td>rasantotomas@gmail.com</td>
      <td>S</td>
      <td>489</td>
    </tr>
  </tbody>
</table>
<p>26000 rows × 23 columns</p>
</div>




```python
pd.DataFrame(list(df3.columns))
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
      <td>a_o</td>
    </tr>
    <tr>
      <th>1</th>
      <td>departamento</td>
    </tr>
    <tr>
      <th>2</th>
      <td>secretaria</td>
    </tr>
    <tr>
      <th>3</th>
      <td>cod_dane_municipio</td>
    </tr>
    <tr>
      <th>4</th>
      <td>municipio</td>
    </tr>
    <tr>
      <th>5</th>
      <td>codigo_dane</td>
    </tr>
    <tr>
      <th>6</th>
      <td>est_id</td>
    </tr>
    <tr>
      <th>7</th>
      <td>nombre_establecimiento</td>
    </tr>
    <tr>
      <th>8</th>
      <td>cte_id_sector</td>
    </tr>
    <tr>
      <th>9</th>
      <td>cte_id_calendario</td>
    </tr>
    <tr>
      <th>10</th>
      <td>sede_id</td>
    </tr>
    <tr>
      <th>11</th>
      <td>codigo_dane_sede</td>
    </tr>
    <tr>
      <th>12</th>
      <td>coordenada_y_sede</td>
    </tr>
    <tr>
      <th>13</th>
      <td>coordenada_x_sede</td>
    </tr>
    <tr>
      <th>14</th>
      <td>nombre_sede</td>
    </tr>
    <tr>
      <th>15</th>
      <td>zona</td>
    </tr>
    <tr>
      <th>16</th>
      <td>direccion</td>
    </tr>
    <tr>
      <th>17</th>
      <td>barrio_vereda</td>
    </tr>
    <tr>
      <th>18</th>
      <td>telefono</td>
    </tr>
    <tr>
      <th>19</th>
      <td>fax</td>
    </tr>
    <tr>
      <th>20</th>
      <td>email</td>
    </tr>
    <tr>
      <th>21</th>
      <td>principal</td>
    </tr>
    <tr>
      <th>22</th>
      <td>total_matricula</td>
    </tr>
  </tbody>
</table>
</div>




```python
#Filtro para las columnas de intéres
df3_coor = df3.iloc[:,[3,11,12,13,14]]
df3_coor.shape
```




    (26000, 5)




```python
df3_coor
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
      <th>cod_dane_municipio</th>
      <th>codigo_dane_sede</th>
      <th>coordenada_y_sede</th>
      <th>coordenada_x_sede</th>
      <th>nombre_sede</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>18150</td>
      <td>218150001809</td>
      <td>0.50</td>
      <td>-74.18</td>
      <td>BUENAVISTA</td>
    </tr>
    <tr>
      <th>1</th>
      <td>18150</td>
      <td>218150001884</td>
      <td>1.28</td>
      <td>-74.66</td>
      <td>SANTA ELENA</td>
    </tr>
    <tr>
      <th>2</th>
      <td>18150</td>
      <td>218150002015</td>
      <td>0.41</td>
      <td>-74.29</td>
      <td>CAÑO NEGRO</td>
    </tr>
    <tr>
      <th>3</th>
      <td>18150</td>
      <td>218150002031</td>
      <td>0.26</td>
      <td>-74.07</td>
      <td>CAÑO SANTO DOMINGO</td>
    </tr>
    <tr>
      <th>4</th>
      <td>18150</td>
      <td>218150002104</td>
      <td>0.47</td>
      <td>-74.17</td>
      <td>SIMON BOLIVAR</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>25995</th>
      <td>76001</td>
      <td>176001005091</td>
      <td>3.46</td>
      <td>-76.51</td>
      <td>INSTITUCION EDUCATIVA REPUBLICA DE ISRAEL</td>
    </tr>
    <tr>
      <th>25996</th>
      <td>76001</td>
      <td>176001006453</td>
      <td>3.46</td>
      <td>-76.52</td>
      <td>JORGE ISAACS</td>
    </tr>
    <tr>
      <th>25997</th>
      <td>76001</td>
      <td>176001003322</td>
      <td>3.47</td>
      <td>-76.52</td>
      <td>MANUELA BELTRAN</td>
    </tr>
    <tr>
      <th>25998</th>
      <td>76001</td>
      <td>176001004132</td>
      <td>3.47</td>
      <td>-76.52</td>
      <td>SANTO TOMAS DE AQUINO</td>
    </tr>
    <tr>
      <th>25999</th>
      <td>76001</td>
      <td>176001040079</td>
      <td>3.47</td>
      <td>-76.52</td>
      <td>INSTITUCION EDUCATIVA SANTO TOMAS  CASD</td>
    </tr>
  </tbody>
</table>
<p>26000 rows × 5 columns</p>
</div>




```python
#Creamos columna que concatena nombre con codigo 
df3_coor['COLE_NOMBRE_SEDE-COLE_COD_MCPIO_UBICACION'] = df3_coor['nombre_sede'] + ' - ' +df3_coor['cod_dane_municipio']
df3_coor
```

    <ipython-input-31-8be5e9d049a2>:2: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      df3_coor['COLE_NOMBRE_SEDE-COLE_COD_MCPIO_UBICACION'] = df3_coor['nombre_sede'] + ' - ' +df3_coor['cod_dane_municipio']





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
      <th>cod_dane_municipio</th>
      <th>codigo_dane_sede</th>
      <th>coordenada_y_sede</th>
      <th>coordenada_x_sede</th>
      <th>nombre_sede</th>
      <th>COLE_NOMBRE_SEDE-COLE_COD_MCPIO_UBICACION</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>18150</td>
      <td>218150001809</td>
      <td>0.50</td>
      <td>-74.18</td>
      <td>BUENAVISTA</td>
      <td>BUENAVISTA - 18150</td>
    </tr>
    <tr>
      <th>1</th>
      <td>18150</td>
      <td>218150001884</td>
      <td>1.28</td>
      <td>-74.66</td>
      <td>SANTA ELENA</td>
      <td>SANTA ELENA - 18150</td>
    </tr>
    <tr>
      <th>2</th>
      <td>18150</td>
      <td>218150002015</td>
      <td>0.41</td>
      <td>-74.29</td>
      <td>CAÑO NEGRO</td>
      <td>CAÑO NEGRO - 18150</td>
    </tr>
    <tr>
      <th>3</th>
      <td>18150</td>
      <td>218150002031</td>
      <td>0.26</td>
      <td>-74.07</td>
      <td>CAÑO SANTO DOMINGO</td>
      <td>CAÑO SANTO DOMINGO - 18150</td>
    </tr>
    <tr>
      <th>4</th>
      <td>18150</td>
      <td>218150002104</td>
      <td>0.47</td>
      <td>-74.17</td>
      <td>SIMON BOLIVAR</td>
      <td>SIMON BOLIVAR - 18150</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>25995</th>
      <td>76001</td>
      <td>176001005091</td>
      <td>3.46</td>
      <td>-76.51</td>
      <td>INSTITUCION EDUCATIVA REPUBLICA DE ISRAEL</td>
      <td>INSTITUCION EDUCATIVA REPUBLICA DE ISRAEL - 76001</td>
    </tr>
    <tr>
      <th>25996</th>
      <td>76001</td>
      <td>176001006453</td>
      <td>3.46</td>
      <td>-76.52</td>
      <td>JORGE ISAACS</td>
      <td>JORGE ISAACS - 76001</td>
    </tr>
    <tr>
      <th>25997</th>
      <td>76001</td>
      <td>176001003322</td>
      <td>3.47</td>
      <td>-76.52</td>
      <td>MANUELA BELTRAN</td>
      <td>MANUELA BELTRAN - 76001</td>
    </tr>
    <tr>
      <th>25998</th>
      <td>76001</td>
      <td>176001004132</td>
      <td>3.47</td>
      <td>-76.52</td>
      <td>SANTO TOMAS DE AQUINO</td>
      <td>SANTO TOMAS DE AQUINO - 76001</td>
    </tr>
    <tr>
      <th>25999</th>
      <td>76001</td>
      <td>176001040079</td>
      <td>3.47</td>
      <td>-76.52</td>
      <td>INSTITUCION EDUCATIVA SANTO TOMAS  CASD</td>
      <td>INSTITUCION EDUCATIVA SANTO TOMAS  CASD - 76001</td>
    </tr>
  </tbody>
</table>
<p>26000 rows × 6 columns</p>
</div>




```python
#Se rervisa el tipo de las columnas
df3_coor.dtypes
```




    cod_dane_municipio                           object
    codigo_dane_sede                             object
    coordenada_y_sede                            object
    coordenada_x_sede                            object
    nombre_sede                                  object
    COLE_NOMBRE_SEDE-COLE_COD_MCPIO_UBICACION    object
    dtype: object




```python
#Se cambia el tipo de la columna de codigo para realizar el merge
df3_coor['cod_dane_municipio'] = pd.to_numeric(df3_coor['cod_dane_municipio'])
```

    <ipython-input-33-9eb5deca2201>:2: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      df3_coor['cod_dane_municipio'] = pd.to_numeric(df3_coor['cod_dane_municipio'])



```python
df3_coor.dtypes
```




    cod_dane_municipio                            int64
    codigo_dane_sede                             object
    coordenada_y_sede                            object
    coordenada_x_sede                            object
    nombre_sede                                  object
    COLE_NOMBRE_SEDE-COLE_COD_MCPIO_UBICACION    object
    dtype: object




```python
#Se. revisan duplicados 
df3_coor.duplicated(subset=['COLE_NOMBRE_SEDE-COLE_COD_MCPIO_UBICACION']).sum()
```




    95




```python
#Se eliminan los duplicados 
df3_coor.drop_duplicates(subset=['COLE_NOMBRE_SEDE-COLE_COD_MCPIO_UBICACION'], inplace=True)
```

    <ipython-input-36-d796857e9836>:2: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      df3_coor.drop_duplicates(subset=['COLE_NOMBRE_SEDE-COLE_COD_MCPIO_UBICACION'], inplace=True)



```python
df3_coor.duplicated(subset=['COLE_NOMBRE_SEDE-COLE_COD_MCPIO_UBICACION']).sum()
```




    0



# Merge Final


```python
#Se realiza el merge entre la base ICFES_KIOSCOS y Coordenadas
b = pd.merge(a, df3_coor, right_on = ["nombre_sede","cod_dane_municipio"], 
             left_on = ["COLE_NOMBRE_SEDE", "COLE_COD_MCPIO_UBICACION"], 
             how = 'left')
```


```python
b.to_csv('~/Downloads/completa.csv', index = False)
```

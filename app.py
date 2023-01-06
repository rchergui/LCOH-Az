import pandas as pd                         ## Manipulation des tables de données et listes
import numpy as np    
import os
import json
import matplotlib.pyplot as plt


from scipy.optimize import minimize         ## solver Objective function

import datetime as dt                       ## Index dataframe to date+houar

import plotly.express as px                 ## Plot figures
import seaborn as sns                       ## Style Figures

import dash                                        ## Application Interface
from jupyter_dash import JupyterDash
from dash import dcc
from dash import html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
import statistics
import math
import itertools

##Input(AEP file_name and directory)

#Lecture Data and reset index to "Date + Time"
def preparation__data_form(file_name):
    time_list = list()
    date_list = list()
    data = pd.read_csv(file_name , delimiter = "\t") 
    time_date = data['Date/Time'].apply(lambda x : x.split(" "))
    for i in range(time_date.size) : 
        time_list.append(time_date[i][1])
        date_list.append(time_date[i][0])
    #Drop Data/Time column and add two more columns Date and Time
    data = data.drop('Date/Time', axis= 1)
    data['Date'],data['Time'] = date_list,time_list
    #Rearrange the order of the columns 
    data = data[['Date','Time',data.columns[0]]]
    return data

AEP_Wind = preparation__data_form(r"Prod Wind 500MW 1996-2020.txt")
AEP_PV = preparation__data_form(r"Prod PV 300MW 1996-2020.txt")
AEP_Data = AEP_Wind
AEP_Data["Prod PV 300MW"] = AEP_PV["Prod PV 300MW"]
AEP_Data['Prod PV 300MW'] = AEP_Data['Prod PV 300MW'].apply(lambda x : float(x.replace(',','.') if ',' in x else float(x)))
AEP_Data.index=pd.to_datetime(AEP_Data['Date']+" "+AEP_Data['Time'])
AEP_Data

##Output (AEP Wind_PV dataframe)
################################

def find_cumul(L):
  i = 0
  LL = {}
  while i < len(L)-1:
    j = i+1
    while L[j]<0 and L[i]<0 and j<len(L):
      LL[i] = {"idx":str(j - i + 1), "total":str(sum(L[i:j+1]))}
      j = j+1
      if j==len(L): break
    i = j
  return LL

def find_cumul_with_pandas(L_, filepath):
  with open(filepath,"w") as f:
    data = find_cumul(L_)
    json.dump(data, f)
  return pd.read_json(filepath, orient="index")



L = np.array([0,1,0,8, 1,2,5,3,2,0,2,2,5,1,2]) - 3
find_cumul_with_pandas(L, "rr.json")

##Output : [LCOH_optimal,  Pw,  Ppv,  C_elec,  Pbess,  total_H2,  total_baseload]

def get_min_LCOH(capex_w,opex_w,minpw,maxpw,
                     capex_pv,opex_pv,minppv,maxppv,
                     capex_elec,opex_elec,conso_spec,minpelec,maxpelec,
                     wacc,QQT_H2_lim,q_nh3):  ##Input

    use_constraints = False
    globals()["list_obj"]=[]
    
    ##quantité_ammoniac
    n=8760
    M0=q_nh3
    M0=M0*1000/8760
    Mn=M0*(100/22.35)*(1+0.2235*n)
    M_h2=0.15*Mn
    M_n2=0.74*Mn

    #QQT_H2_lim =QQT_H2_lim*1000
    QQT_H2_lim =M_h2*1
    
    ##fonction stockage hydro
    def hydro_stock(var):
        return (var[0] + var[1])  * float(1000) / float(conso_spec)   -QQT_H2_lim
    
    def hydro_stock_v2(var):
        return (min(var[0] + var[1],float(var[2]))  * float(1000) / float(conso_spec) )  -QQT_H2_lim
    
    def hydro_annuel(var):
        list_puissance_temp = AEP_Data['Prod W 500MW']*(var[0]/500) + AEP_Data['Prod PV 300MW']*(var[1]/300)
        list_puissance_temp[list_puissance_temp>var[2]] = var[2]
        val= (1000 / conso_spec)*(list_puissance_temp).sum()
        val =  (val*(1/25) - QQT_H2_lim)/QQT_H2_lim
        return val
    

    def LCOH(mix_values):       ### mix_values : liste = [Pw, Ppv, C_elec, Pbess]                                                                            
        ### AEP_Data : table de données regroupant tous les données de production
        ### Initialisation colonnes dataframe
        AEP_Data["Energy Wind ratio"] = AEP_Data['Prod W 500MW'] / float(500)     
        AEP_Data["Energy PV ratio"] = AEP_Data['Prod PV 300MW'] / float(300)
        AEP_Data["Energy Wind"] = ""
        AEP_Data["Energy PV"] = ""
        AEP_Data["Energy Hybrid"] = ""
        AEP_Data["Energy Bridée"] = ""
        AEP_Data["Energy Electrolysis (MWh)"] = ""
        AEP_Data["Hydrogen Production (Kg H2)"] = ""

        ### Calcul production Wind et PV à la base de ratio horaire de 500MW et 300 MW resp.
        AEP_Data["Energy Wind"] = AEP_Data["Energy Wind ratio"] * float(mix_values[0])       
        AEP_Data["Energy PV"] = AEP_Data["Energy PV ratio"] * float(mix_values[1]) 
        
        ### Energy Hybrid = Wind + PV
        AEP_Data["Energy Hybrid"] = AEP_Data["Energy Wind"] + AEP_Data["Energy PV"]          
        pelec= mix_values[2]
        
        ### Production_H2
        AEP_Data.at[(AEP_Data["Energy Hybrid"] > float(pelec)) , "Energy Electrolysis (MWh)"] =  float(pelec)               
        AEP_Data.at[(AEP_Data["Energy Hybrid"] <= float(pelec)) , "Energy Electrolysis (MWh)"] =  AEP_Data["Energy Hybrid"]
        AEP_Data["Hydrogen Production (Kg H2)"] = (AEP_Data["Energy Electrolysis (MWh)"] * float(1000)) / float(conso_spec)
        AEP_Data["Energy Bridée"] = AEP_Data["Energy Hybrid"] - AEP_Data["Energy Electrolysis (MWh)"]
        
        ### nombre de cycle électrolyse pour remplacement
        life_time_elec = AEP_Data["Energy Electrolysis (MWh)"].apply(lambda x: True if x != 0 else False)    
        nbr_cycle = len(life_time_elec[life_time_elec == True].index)
        stack_lifetime = 75000
        T=25

        ### Total Cost
        def NPCost():               
            NPC = (capex_elec * pelec) + (capex_w * mix_values[0]) + (capex_pv *mix_values[1])
            for i in range (T):
                NPC += ((opex_elec * pelec) + (opex_w * mix_values[0]) + (opex_pv * mix_values[1])) / ((1 + wacc)**i)
            
            ####stack replacement
            if stack_lifetime < nbr_cycle < 2*stack_lifetime:        ### remplacer stack électrolyse
                NPC += ((0.4*capex_elec* pelec) / ((1 + wacc)**7))
            elif 2*stack_lifetime < nbr_cycle < 3*stack_lifetime:
                NPC += (((0.4*capex_elec* pelec) / ((1 + wacc)**7)) + ((0.4*capex_elec* pelec) / ((1 + wacc)**14)))

            elif 3*stack_lifetime < nbr_cycle < 4*stack_lifetime:
                NPC += (((0.4*capex_elec* pelec) / ((1 + wacc)**7)) + ((0.4*capex_elec* pelec) / ((1 + wacc)**14)) 
                        + ((0.4*capex_elec* pelec) / ((1 + wacc)**21)))
            return NPC 
        
        ## Energy Baseload
        def NPV_energyH2():        
            AEP = 0
            agregate_=AEP_Data["Hydrogen Production (Kg H2)"].resample(rule='A').sum()
            for i in range(T):
                AEP += (agregate_[i] / ((1+wacc)**i)) ##agrégation des AEP pour avoir des somme sur une année.
            return AEP,agregate_
        
        globals()['npv_ENERGY']=NPV_energyH2()
        
        ### Fonction objective
        return_vav = ((NPCost())*1e4) / ( globals()['npv_ENERGY'][0])
        globals()["list_obj"].append(return_vav)
        return return_vav
        ##END LCOH function    
    
    
    def const_squared(mix_values):
        return hydro_stock(mix_values)**2 if use_constraints else hydro_annuel(mix_values)**2
        
    def objective(mix_values):
        return LCOH(mix_values) if use_constraints else LCOH(mix_values) + const_squared(mix_values)
    

    ##Paramètres d'optimisation : contraintes, valeurs initiales  
    ######################################################################################################################################################################""

    # initial guesses au hasard
    mix_values0 = np.zeros(3)
    mix_values0[0] = 700     ##Pw
    mix_values0[1] = 600      ##Ppv
    mix_values0[2] =300
    # show initial objective
    print('Initial Objective: ' + str(LCOH(mix_values0)/1e4))

   # domaine de définition
    b1=(minpw,maxpw)  ##Pw 
    b2=(minppv,maxppv)  ##Ppv
    b3=(minpelec,maxpelec)
    bnds = (b1,b2,b3)
    
    

    conss={'type':'eq','fun':hydro_stock} if use_constraints == True else {'type':'eq','fun':hydro_annuel}

    ##Appel de la fonction d'optimisation
    if use_constraints:  ##SLSQP : Sequential Least SQuare Programming 
        solution = minimize(objective,mix_values0,method='SLSQP',bounds=bnds, constraints=conss,options={'maxiter': 200, 'ftol': 1e-5})#options={'maxiter': 200, 'ftol': 1e-3})   ##options : nbr itérations maximale et précision
    else:
        solution = minimize(objective,mix_values0,method='SLSQP',bounds=bnds, constraints=conss, options={'maxiter': 200, 'ftol': 1e-5})


    
    mix_values = solution.x    ### solution contient le détail de l'optimisation
    fct_print = lambda *x: print(*x, file=open("amm.txt","a"))
    open("amm.txt","w")
    fct_print(solution)    ### True or False

    # print solution
    fct_print('LCOH : ' + str(LCOH(mix_values)/1e4))
    fct_print("conss = ",hydro_annuel(mix_values))
    fct_print('Pw = ' + str(mix_values[0]) + '(MW)' )
    fct_print('Ppv = ' + str(mix_values[1])+ '(MW)')
    fct_print('pelec=' + str(mix_values[2])+'(MW)')
    
    prod_jour=AEP_Data["Hydrogen Production (Kg H2)"].resample(rule='D').sum()
    fct_print('Production minimale (Kg)', min(prod_jour))
    fct_print('Production maximale (Kg)', max(prod_jour))
    total_H2 = AEP_Data["Hydrogen Production (Kg H2)"].sum() / 25
    energy_lost = (AEP_Data["Energy Bridée"].sum() * 100) / AEP_Data["Energy Hybrid"].sum()
    
    prod_diff= prod_jour - M_h2/365
    c=0
    prod_neg=[]
    #PROD_NEGG=[]
    for i in prod_diff:
      if  i<=0:
        prod_neg.append(i)
        c=c+1
    #fct_print(prod_neg)
    fct_print('Manque maximale (Kg)', min(prod_neg))
    fct_print('Manque minimale (Kg)', max(prod_neg))

    find_cumul_with_pandas(prod_diff,"temp.json")


    prod_neg_mean= statistics.mean(prod_neg) #valeur negative
    fct_print('Moyenne', prod_neg_mean)
    prod_neg_st_dev=statistics.pstdev(prod_neg)
    fct_print('Ecart type', prod_neg_st_dev)
    Dim_stock= 2*prod_neg_st_dev-prod_neg_mean #Stockage avec un taux de confiance de 97,5%/68/95
    fct_print('Stockage_H2_95: ',Dim_stock)
    
    import json
    with open("prod_neg.json", "w") as f:
      json.dump(prod_neg, f, indent=4)

    
    #prod_neg = pd.Series(prod_neg)   
    
    #prod_neg=prod_neg.resample(rule='A').sum()
    #fct_print(prod_neg)
        


    c=100*c/len(prod_diff)
    fct_print("Pourcentage d'indisponibilité = ", c)


    #volume_stockage
    m=total_H2
    M=2
    R=8.3144
    T=343.15
    P=300000
    V=(m*R*T)/(M*P)

   
    
    fct_print('H2 annual production = ' + str(total_H2) + 'kg')
    fct_print("Volume =" + str(V)+'m^3')
    fct_print('Energy bridée =' + str(energy_lost) + '%' )

    lcoh_ret = LCOH(mix_values)/1e4
    return [lcoh_ret,mix_values[0],mix_values[1],mix_values[2],total_H2,energy_lost,M_h2/365,M_n2/365]
	
dash_app = JupyterDash(external_stylesheets=[dbc.themes.MINTY])
dash_app.layout = dbc.Container(
    [
        dbc.Row([
            dbc.Col([
                html.H2("LCOH Optimization - H2 Production", style={'color' : 'navy', 'fontSize' : 38, 'font-weight': '600'}),
                html.H5("version finale"),
            ], width=True),
        ], align="end"),
        html.Hr(),
        
#         dbc.Row([dbc.Col([html.Div([dcc.Upload(dbc.Button('Upload File',id="upload",color="info"),
#                         loading_state={'is_loading':True})],),
#                          ]),]),     
    
        html.Hr(),
        dbc.Row([
            dbc.Col([
                html.Div([
                    html.H5("Wind Parameters", style={'color' : 'royalblue', 'fontSize' : 25, 'font-weight': '400','font-family': 'sans-serif'}),
                    html.P("CapEx [$/MW]:", style={'color' : 'black', 'fontSize' : 20, 'font-weight': '400'}),
                    dbc.Input(id='capex_w',value=1000000 , placeholder="hint (1000000 ; 1200000", type="number",min=0),
                    html.P("OpEx [$/MW/year]:", style={'color' : 'black', 'fontSize' : 20, 'font-weight': '400'}),
                    dbc.Input(id='opex_w',value=23200,placeholder="hint (2.32% Capex)", type="number",min=0),
                    html.P("Min Bound Pw", style={'color' : 'black', 'fontSize' : 20, 'font-weight': '400'}),
                    dbc.Input(id='minpw',value=0,placeholder="minimum installed capacity", type="number",min=0),
                    html.P("Max Bound Pw", style={'color' : 'black', 'fontSize' : 20, 'font-weight': '400'}),
                    dbc.Input(id='maxpw',value=2000,placeholder="maximum installed capacity", type="number",min=0),
                    html.P("Quantité annuelle NH3 visée(t)", style={'color' : 'black', 'fontSize' : 20, 'font-weight': '400'}),
                    dbc.Input(id='q_nh3',value=85377,placeholder="maximum installed capacity", type="number",min=0),
                ]),
            ]),
            dbc.Col([
                html.Div([
                    html.H5("Solar PV Parameters", style={'color' : 'royalblue', 'fontSize' : 25, 'font-weight': '400','font-family': 'sans-serif'}),
                    html.P("CapEx [$/MW]:", style={'color' : 'black', 'fontSize' : 20, 'font-weight': '400'}),
                    dbc.Input(id='capex_pv',value=617000 , placeholder="hint (617000)" ,type="number",min=0),
                    html.P("OpEx [$/MW/year]:", style={'color' : 'black', 'fontSize' : 20, 'font-weight': '400'}),
                    dbc.Input(id='opex_pv',value=17030 ,placeholder="hint (2.76% Capex)", type="number",min=0),
                    html.P("Min Bound Ppv", style={'color' : 'black', 'fontSize' : 20, 'font-weight': '400'}),
                    dbc.Input(id='minppv',value=0,placeholder="minimum installed capacity", type="number",min=0),
                    html.P("Max Bound Ppv", style={'color' : 'black', 'fontSize' : 20, 'font-weight': '400'}),
                    dbc.Input(id='maxppv',value=2000,placeholder="maximum installed capacity", type="number",min=0), 
                    html.P("Quantité annuelle visée(t)", style={'color' : 'black', 'fontSize' : 20, 'font-weight': '400'}),
                    dbc.Input(id='QQT_H2_lim',value=80000,placeholder="maximum installed capacity", type="number",min=0),
                    
                    #html.P("Quantité_horaire", style={'color' : 'black', 'fontSize' : 20, 'font-weight': '400'}),
                    #dbc.Input(id='QQT_H2_lim',value=14000,placeholder="hint (14000)", type="number",min=0),
                ]),
            ]),
            dbc.Col([
                html.Div([
                    html.H5("Electrolysis Parameters", style={'color' : 'royalblue', 'fontSize' : 25, 'font-weight': '400','font-family': 'sans-serif'}),
                    html.P("CapEx [$/MW]:", style={'color' : 'black', 'fontSize' : 20, 'font-weight': '400'}),
                    dbc.Input(id='capex_elec',value=847000, placeholder="hint (700000;600000) + 11% + 10 %, installation/soft cost", type="number",min=0),
                    html.P("OpEx [$/MW/year]:", style={'color' : 'black', 'fontSize' : 20, 'font-weight': '400'}),
                    dbc.Input(id='opex_elec', value=16940,placeholder="hint (2% Capex not including stack replacement)", type="number",min=0),
                    html.P("Specific Electricity Consumption [kWh/kg]", style={'color' : 'black', 'fontSize' : 20, 'font-weight': '400'}),
                    dbc.Input(id='conso_spec',value=52,placeholder="hint (54 ;52 ;50)", type="number",min=33, max=100),
                    html.P("Max Bound Electrolysis", style={'color' : 'black', 'fontSize' : 20, 'font-weight': '400'}),
                    dbc.Input(id='maxpelec',value=4000,placeholder="minimum installed capacity", type="number",min=1),
                    html.P("Min Bound Electrolysis", style={'color' : 'black', 'fontSize' : 20, 'font-weight': '400'}),
                    dbc.Input(id='minpelec',value=0,placeholder="minimum installed capacity", type="number",min=1),
                    #html.P("Quantité_horaire", style={'color' : 'black', 'fontSize' : 20, 'font-weight': '400'}),
                    #dbc.Input(id='QQT_H2_lim',value=14000,placeholder="hint (14000)", type="number",min=1),

                ]),
            ]),
            
        ]),
        html.Hr(),
        dbc.Row([
            dbc.Col([
                html.Div([
                    html.H5("System parameters", style={'color' : 'royalblue', 'fontSize' : 25, 'font-weight': '400','font-family': 'sans-serif'}),
                    html.P("Wacc (0.0X):", style={'color' : 'black', 'fontSize' : 20, 'font-weight': '400'}),
                    dbc.Input(id='wacc',value=0.07,placeholder="hint (0.07)", type="number",min=0),
                    html.Hr(),
                    html.H5("Commands"),
                    dbc.Button("Run Optimization (~2min)", id="opt", size="lg",color="success", style={"margin": "10px"},
                               n_clicks_timestamp='0'),
                    dbc.Button("Stop Optimization",id="stop",color="warning", style={"margin": "10px"},
                               n_clicks_timestamp='0')

                ]),
            ],width=4),
        ]),
        html.Hr(),
        html.Hr(),
        html.H2('Results'),
        dbc.Row([
            dbc.Col([
                html.Div([
                    html.H5("LCOH / mix capacity table"),
                    dbc.Spinner(
                    html.P(id='output'),
                    color="success",
                    type="grow")
                    ])
            ]),
            dbc.Col([
                html.Div([
                 html.H5("Annual Production"),  
                 dbc.Spinner(   
                 dcc.Graph(id='graph-output', figure={}),color="success",type="grow")
                ])
            ])
        ]),
        html.Hr(),
    ],
    fluid=True)

##Display results in Dataframe
def make_table(dataframe):
    return dbc.Table.from_dataframe(
        dataframe,
        bordered=True,
        hover=True,
        responsive=True,
        striped=True,
        color='light',
        style={

        }
    )
T=25
######################################## Call Back


@dash_app.callback(
    [Output('output', 'children'),Output('graph-output', 'figure')],
    [Input("opt", 'n_clicks_timestamp'),Input("stop", 'n_clicks_timestamp')],
    [
        State('capex_w', 'value'),
        State('opex_w', 'value'),
        State('minpw', 'value'),
        State('maxpw', 'value'),
        State('capex_pv', 'value'),
        State('opex_pv', 'value'),
        State('minppv', 'value'),
        State('maxppv', 'value'),
        State('capex_elec', 'value'),
        State('opex_elec', 'value'),
        State('conso_spec', 'value'),
        State('minpelec', 'value'),
        State('maxpelec', 'value'),
        State('wacc', 'value'),
        State('QQT_H2_lim', 'value'),
        State('q_nh3', 'value')
    ],
    prevent_initial_call=True
)

def Run_optimization(opt,stop,capex_w,opex_w,minpw,maxpw,
                     capex_pv,opex_pv,minppv,maxppv,
                     capex_elec,opex_elec,conso_spec,minpelec,maxpelec,
                     wacc,QQT_H2_lim,q_nh3):
        try:
            button_pressed = np.argmax(np.array([
                float(opt),
                float(stop),
            ]))
            assert button_pressed is not None
        except:
            button_pressed = 0

        if button_pressed == 0:    

            result=get_min_LCOH(capex_w,opex_w,minpw,maxpw,
                             capex_pv,opex_pv,minppv,maxppv,
                             capex_elec,opex_elec,conso_spec,minpelec,maxpelec,
                             wacc,QQT_H2_lim,q_nh3)

            output = make_table(pd.DataFrame(
                {
                    "Output": [
                        "LCOH ($/kg)",
                        "Capacité éolienne (MW)",
                        "Capacité Solaire PV (MW)",
                        "Capacité électrolyse (MW)",
                        "Curtailment (%)",
                        "Annual quantity (Kg)",
                        "Masse journaliere H2 (Kg)",
                        "Masse journaliere N2 (Kg)"
                    ],
                    "Value" : [
                        round(float(result[0]),3),
                        round(float(result[1]),0),
                        round(float(result[2]),0),
                        round(float(result[3]),0),
                        round(float(result[5]),2),
                        round(float(result[4]),2),
                        round(float(result[6]),2),
                        round(float(result[7]),2)
                    ]
                }
            ))

            test=AEP_Data.drop(['Date','Time'],axis=1)
            test=test.astype(np.float64)
            test=test.resample(rule='A').sum()
            test['Hydrogen Production / 100 (tonnes)'] = test["Hydrogen Production (Kg H2)"] / 100000
            test["Energy Electrolysis (GWh)"] = test["Energy Electrolysis (MWh)"] / 1000
            test["Energy Wind+PV (GWh)"] = test["Energy Hybrid"] / 1000

            fig = px.bar(data_frame=test, 
                       x=test.index.year, 
                       y=['Energy Wind+PV (GWh)',"Energy Electrolysis (GWh)","Hydrogen Production / 100 (tonnes)"],
                       color_discrete_sequence =['goldenrod','orangered','navy'],
                       template="plotly_white",
                       width=1200,
                       title="Annual Production",
                       barmode='group',
            )
            fig.update_layout(width=1200, height=500, bargap=0.3)
            fig.update_traces(width=0.2)
            fig.update_xaxes(ticks="outside",nticks=50,tickwidth=1,tickangle=60,title_text="Year",title_font=dict(color='navy'))
        
        elif button_pressed == 1:
             output = str('Optimization Stoped')
            #  output == None
            #  fig == {}

            
            
        return output, fig


if __name__ == "__main__":
    # Get port and debug mode from environment variables    
    port = os.environ.get('dash_port')
    debug = os.environ.get('dash_debug')=="True"
    dash_app.run_server(debug=debug, host="0.0.0.0", port=port)

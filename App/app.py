from kivymd.app import MDApp
from kivy.logger import Logger ,LOG_LEVELS
# Logger.setLevel(LOG_LEVELS['info'])
from kivy.factory import Factory 
from kivy.lang.builder import Builder
from kivy.properties import ObjectProperty ,NumericProperty  ,StringProperty,BooleanProperty ,DictProperty ,ListProperty
from kivymd.uix.relativelayout import MDRelativeLayout
from kivymd.uix.label import MDLabel
from kivy.uix.label import Label
from kivymd.uix.card import MDCard
from kivy.uix.recycleview import RecycleView
from kivymd.uix.chip import MDChip
from kivymd.uix.tooltip import MDTooltip
from kivymd.uix.boxlayout import MDBoxLayout
from kivy.graphics import Line ,Color
import numpy as np

import pandas as pd
from scipy.optimize import minimize
from datetime import datetime, timedelta
from sklearn.metrics import mean_squared_error

from scipy.optimize import curve_fit
from tqdm import tqdm
from kivy_garden.graph import Graph, MeshLinePlot
from math import sin, cos
from  kivy.uix.modalview import ModalView
from kivymd.uix.snackbar import Snackbar
from kivymd.uix.gridlayout import MDGridLayout
import json 
# Données spécifiques pour le Port San Luis, CA (hauteur d'eau moyenne, amplitudes, fréquences, déphasages)
Z = 0.0  # Hauteur d'eau moyenne
amplitudes = [0.492, 0.149, 0.113, 0.357, 0.002, 0.223, 0.0, 0.001, 0.001, 0.001]  # Amplitudes des 10 premiers constituants
frequences = [28.984104, 30.0, 28.43973, 15.041069, 57.96821, 13.943035, 86.95232, 44.025173, 60.0, 57.423832]  # Fréquences des 10 premiers constituants
dephasages = [296.3, 283.7, 276.0, 94.4, 171.0, 87.3, 0.0, 243.6, 185.1, 118.4]  # Déphasages des 10 premiers constituants (en radian)

from kivymd.uix.filemanager import MDFileManager

from kivymd.uix.picker import MDDatePicker
from kivy.core.window import Window

import os
# os.path.dirname(os.path.abspath(__file__))
from kivy.storage.jsonstore import JsonStore
from kivymd.uix.dialog import MDDialog
from kivymd.uix.button import MDFlatButton

#import clock from kivy
from kivy.clock import Clock

class CountryCard(MDCard):

    index = None
    selected = BooleanProperty(False)
    selectable = BooleanProperty(True)
    text = StringProperty()
    def __init__(self, **kwargs):
        super(CountryCard, self).__init__(**kwargs)





class RVConutry(RecycleView):
    def __init__(self, **kwargs):
        super(RVConutry, self).__init__(**kwargs)
        self.app = MDApp.get_running_app()
        
        self.data = [{'text': str(country)  } for country in self.app.ports]

    def search(self , text="",):

        
        self.data = []
        results = []
        for c in self.app.ports:
            if text.lower() in c.lower():
                results.append(c)
 
        self.data = [{'text': str(country)  } for country in results]

class PortCard(MDCard):

    index = None
    selected = BooleanProperty(False)
    selectable = BooleanProperty(True)
    text = StringProperty()
    country = StringProperty("")
    def __init__(self, **kwargs):
        super(PortCard, self).__init__(**kwargs)

class ConstituantChip(MDChip,MDTooltip):

    index = None
    selected = BooleanProperty(False)
    selectable = BooleanProperty(True)
    text = StringProperty()
    country = StringProperty("")
    def __init__(self, **kwargs):
        super(ConstituantChip, self).__init__(**kwargs)


class Search_dropdown(ModalView):
    datas = ListProperty([])
    selcted =  StringProperty("")
    search_mode = StringProperty("country")
    instance = StringProperty("home_screen")

    def __init__(self, **kwargs):
        super(Search_dropdown, self).__init__(**kwargs)
        self.app = MDApp.get_running_app()


    def set_list_md_icons(self , text="", search=False):

        '''Builds a list of icons for the screen MDIcons.'''
        if self.search_mode == "country":
            self.datas =  list(self.app.ports.keys())
        else:
            

            if self.instance=="home_screen" and self.app.selected_country:
                self.datas = list(self.app.ports[self.app.selected_country])


            elif self.instance=="add_port" and self.app.Add_port.selected_country:
                self.datas = list(self.app.ports[self.app.Add_port.selected_country])

            else:
                self.datas = []

        

        def add_icon_item(country_type):
            self.ids.rv.data.append(
                {
                    "viewclass": "OneLineListItem",
                    "text": country_type,
                    "on_release":lambda x =0 :self.select_country(country_type),
                }
            )

        self.ids.rv.data = []
        for data in self.datas:
            data=str(data)
            if search:

                if text.lower() in data.lower():
                    add_icon_item(data)

            else:
                add_icon_item(data)

    def select_country(self,value):

        if self.instance == "home_screen":
            self.app.select_port(self.search_mode,value)
        
        elif self.instance == "add_port":
            self.app.Add_port.select_port(self.search_mode,value)

        self.selcted = value
        self.dismiss()

    def Oopen(self,search_mode,instance="home_screen"):
        self.search_mode = search_mode
        self.instance = instance
        self.open()
        self.set_list_md_icons()
 
        


class RVPort(RecycleView):
    country = StringProperty("")
    def __init__(self, **kwargs):
        super(RVPort, self).__init__(**kwargs)
        self.app = MDApp.get_running_app()


    def update(self):
        if  not self.country :
            return
        ports = self.app.ports[self.country]

        self.data = [{'country':self.country,'text': str(port)  } for port in ports]

    def search(self , text="",):
        if  not self.country :
            return
        
        self.data = []
        results = []
        ports = self.app.ports[self.country]
        for c in ports:
            if text.lower() in c.lower():
                results.append(c)
 
        self.data = [{'country':self.country,'text': str(country)  } for country in results]
class RVConstituant(RecycleView):
    country = StringProperty("")
    # text2 = StringProperty("")
    def __init__(self, **kwargs):
        super(RVConstituant, self).__init__(**kwargs)
        self.app = MDApp.get_running_app()
        self.data = []

    def update(self):

        if self.app.ports[self.app.Add_port.selected_country][self.app.Add_port.selected_port]['constituants'] :
            self.data = [{"constituant_name" : str(name),"amplitude_optimized": str(round(values[0],5)),"phase_optimized": str(round(values[1],5)), } for name , values in self.app.ports[self.app.Add_port.selected_country][self.app.Add_port.selected_port]['constituants'].items()]
        else:
            self.data = []
class RVData(RecycleView):
    country = StringProperty("")
    # text2 = StringProperty("")
    def __init__(self, **kwargs):
        super(RVData, self).__init__(**kwargs)
        self.app = MDApp.get_running_app()



        self.data = []

    def update(self):

        if self.app.ports[self.app.Add_port.selected_country][self.app.Add_port.selected_port]['data'] :

            self.data = [{"date" : str(date),"time": str(values[0]),"predicted": str(values[1]),"verified": str(round(values[2],2))} for date , values in self.app.ports[self.app.Add_port.selected_country][self.app.Add_port.selected_port]['constituants'].items()]
        else:
            self.data = []




class GraphModel_1(MDRelativeLayout):
    grid_nombre = NumericProperty(24)
    prediction = ListProperty([])
    graphs = ListProperty([])
    predicted_optimized_plot = ObjectProperty(MeshLinePlot(color=[.2, .1, .5, 1]))

    def __init__(self, **kwargs):
        super(GraphModel_1, self).__init__(**kwargs)
        # self.prediction = np.arange(0, 24, 0.1)
    
        # self.new_graph()

        self.clear_widgets()
        self.graph = Graph(xlabel="", ylabel="y", x_ticks_major=1, y_ticks_major=1,
                           y_grid_label=True, x_grid_label=True, x_grid=False, y_grid=True,
                           xmin=0, xmax=23, ymin=-1, ymax=1, draw_border=False,label_options={'color': (0, 0, 0, 1),},)

        self.add_widget(self.graph)
        self.graph.add_plot(self.predicted_optimized_plot)
        self.bind(pos=self.updateDisplay)
        self.bind(size=self.updateDisplay)
        self.bind(prediction=self.updateDisplay)

    def add_plot(self,plot):
        self.graph.add_plot(plot)
        self.graphs.append(plot)


    def reset_graph(self):
        for plot in self.graphs:
            self.graph.remove_plot(plot)
        self.graphs = []




    def updateDisplay(self, *args):
        

        self.graph.size = (self.width, self.height)

        # if self.prediction:
        #     # self.plot.points = [(x, sin(x / 10.)) for x in range(0, 101)]
        #     # print(self.prediction)
        #     self.plot.points = [(i,  p) for i,p in enumerate(self.prediction)]

 
            
    def update_graph(self, plot):

        self.graph.xmax = len(plot) -1
        self.graph.x_ticks_major = 1
        min_t =int(min(plot))
        max_t =int(max(plot)) 
        self.graph.ymin = min_t -1
        self.graph.ymax = max_t +1




class Add_port(MDBoxLayout):
    selected_port = StringProperty("")
    selected_country = StringProperty("")

    def init(self):
        self.app = MDApp.get_running_app()
        countrys = list(self.app.ports.keys())

        self.selected_country = countrys[0] if len(countrys) else ""



        if self.selected_country:
            ports = list(self.app.ports[self.selected_country])
            print(self.selected_country,"ports",ports)
            self.selected_port = ports[0] if len(ports) else ""
        else:
            self.selected_port = ""

    def select_port(self, search_mode, value):
        if search_mode == "country":
            self.selected_country = value
            ports = list(self.app.ports[self.selected_country])
            self.selected_port = ports[0] if len(ports) else ""
        

        elif search_mode == "port":
            self.selected_port = value
            # if self.app.ports[self.selected_country][self.selected_port]['constituants']:
                # self.load_constituant()
            self.ids.rv_constituant.update()
            # self.ids.rv_data.update()


class Add_country_dialog(MDBoxLayout):
    pass
class Add_port_dialog(MDBoxLayout):
    pass


class Constituant_MDGridLayout(MDGridLayout):
    constituant_name = StringProperty("")
    phase_optimized= StringProperty("")
    amplitude_optimized= StringProperty("")
    speed_optimized= StringProperty("")
    
    def __init__(self,**kwargs):
        super(Constituant_MDGridLayout, self).__init__(**kwargs)
class Data_MDGridLayout(MDGridLayout):
    date = StringProperty("")
    time= StringProperty("")
    predicted= StringProperty("")
    verified= StringProperty("")
    def __init__(self,**kwargs):
        super(Data_MDGridLayout, self).__init__(**kwargs)

class Single_maree(MDGridLayout):
    designation = StringProperty("")
    heure= StringProperty("")
    hauteur= StringProperty("")
    index= NumericProperty(1)

class Export_DAY(ModalView):
    date = StringProperty("")

class MainApp(MDApp):
    
    start_date = StringProperty("2023-10-01")#111
    end_date = StringProperty("2023-10-01")#111
    model_1 = ObjectProperty(None)
    portScreen = ObjectProperty(None)
    countryScreen = ObjectProperty(None)
    filterCOnstituant = ListProperty([])
    search_dropdown = ObjectProperty(None,allownone=True)
    selcted =StringProperty("")   
    selected_port = StringProperty("")
    selected_country = StringProperty("")
    date_mode= StringProperty("from") 
    start_date = StringProperty("2023-10-01")  
    tide_data = ObjectProperty(None)
    store = ObjectProperty(JsonStore('data.json'))
    ModalAdd= ObjectProperty(None)
    Add_port= ObjectProperty(None)
    dialog_port = None
    dialog_country = None


    constituants_description = DictProperty({"M2": 	"Principal lunar semidiurnal constituent",                  
                                    "S2":"Principal solar semidiurnal constituent",
                                    "N2":"Larger lunar elliptic semidiurnal constituent",
                                    "K1":"Lunar diurnal constituent",
                                    "M4":"Shallow water overtides of principal lunar constituent",
                                    "O1":"Lunar diurnal constituent",
                                    "M6":"Shallow water overtides of principal lunar constituent",
                                    "MK3":"Shallow water terdiurnal",
                                    "S4":"Shallow water overtides of principal solar constituent",
                                    "MN4":"Shallow water quarter diurnal constituent",
                                    "NU2":"Larger lunar evectional constituent",
                                    "S6": "Shallow water overtides of principal solar constituent",
                                    "MU2":"Variational constituent",
                                    "2N2":"Lunar elliptical semidiurnal second-order constituent",
                                    "OO1":"Lunar diurnal",
                                    "S1":"Solar diurnal constituent",
                                    "M1":"Smaller lunar elliptic diurnal constituent",
                                    "J1":"Smaller lunar elliptic diurnal constituent",
                                    "MM":"Lunar monthly constituent",
                                    "SSA":"Solar semiannual constituent",
                                    "SA":"Solar annual constituent",
                                    "MSF":"Lunisolar synodic fortnightly constituent",
                                    "MF":"Lunisolar fortnightly constituent",
                                    "RHO":"Larger lunar evectional diurnal constituent",
                                    "Q1":"Larger lunar elliptic diurnal constituent",
                                    "T2":"Larger solar elliptic constituent",
                                    "R2":"Smaller solar elliptic constituent",
                                    "2Q1":"Larger elliptic diurnal",
                                    "P1":"Solar diurnal constituent",
                                    "M3":"Lunar terdiurnal constituent",
                                    "L2":"Smaller lunar elliptic semidiurnal constituent",
                                    "2MK3":"Shallow water terdiurnal constituent",
                                    "K2":"Lunisolar semidiurnal constituent",
                                    "M8": "Shallow water eighth diurnal constituent",
                                    "MS4":"Shallow water quarter diurnal constituent",
                                    "2SM2":"Shallow water semidiurnal constituent",
                                    "LAM2":"Smaller lunar evectional constituent",})

    list_training_constituants = ListProperty(["M2","S2","N2","K1","O1","P1","M4"])
  
    ports_old = DictProperty({
                        "Etats-Unis": {
                            "Los Angeles" : [],
                            "New York" : [],
                            "San Francisco" : [],
                            "San Luis" : {
                            "data":{},
                            "constituantsd": {
                                    "M2": [1.61,	296.3,	28.984104],                  
                                    "S2":[0.49,	283.7,	30.0,     ],
                                    "N2":[0.37,	276.0,	28.43973],
                                    "K1":[1.17,	94.4,	15.041069],
                                    "M4":[ 0.01,	171.0,	57.96821],
                                    "O1":[0.73,	87.3,	13.943035],
                                    "M6":[0.0,	0.0	 ,   86.95232],
                                    "MK3":[0.0,	243.6,	44.025173],
                                    "S4":[0.0,	185.1,	60.0,     ],
                                    "MN4":[0.0,	118.4,	57.423832],
                                    "NU2":[0.07,	282.4,	28.512583],
                                    "S6":[0.0,	43.6,	90.0,     ],
                                    "MU2":[0.05,	239.6,	27.968208],
                                    "2N2":[0.05,	250.9,	27.895355],
                                    "OO1":[0.04,	118.5,	16.139101],
                                    "S1":[0.02,	206.7,	15.0,     ],
                                    "M1":[0.04,	114.2,	14.496694],
                                    "J1":[0.07,	102.2,	15.5854435],
                                    "MM":[0.0,	0.0	 ,   0.5443747],
                                    "SSA":[0.0,	0.0	 ,   0.0821373],
                                    "SA":[0.23,	190.2,	0.0410686],
                                    "MSF":[0.0,	0.0	 ,   1.0158958],
                                    "MF":[0.03,	120.5,	1.0980331],
                                    "RHO":[0.02,	85.2,	13.471515],
                                    "Q1":[0.13,	84.3,	13.398661],
                                    "T2":[0.03,	271.2,	29.958933],
                                    "R2":[0.0,	0.0	 ,   30.041067],
                                    "2Q1":[0.02,	91.5,	12.854286],
                                    "P1":[0.37,	92.5,	14.958931],
                                    "M3":[0.01,	13.1,	43.47616],
                                    "L2":[0.03,	293.0,	29.528479],
                                    "2MK3":[0.0,	233.4,	42.92714],
                                    "K2":[    0.14,	275.7,	30.082138],
                                    "M8":[    0.0,	0.0	 ,   115.93642],
                                    "MS4":[   0.0,	162.5,	58.984104],
                                    "2SM2":[  0.01,	104.1,	31.015896],
                                    "LAM2":[  0.01,	305.9,	29.455626],},
                            },
                            }
                        ,})
    ports = DictProperty({
                        "Etats-Unis": {
             
                            "San Luis" : {
                            "constituants": {
                                'M2':[0, 0],
                                'S2':[0, 0],
                                'N2':[0, 0],
                                'K1':[0, 0],
                                'O1':[0, 0],
                                'P1':[0, 0],
                                'M4':[0, 0],
                            },
                            "RMSE": 0,
                            "msl": 0,
                  
                            },
                            }
                        ,})
    speeds = DictProperty({
                        "M2": 28.984104,                  
                        "S2": 30.0,
                        "N2": 28.43973,
                        "K1": 15.041069,
                        "O1": 13.943035,
                        "P1": 14.958931,
                        "M4": 57.96821,
                        })
    # Combine date and time into one datetime column (if necessary) and convert to seconds since epoch
    epoch = ObjectProperty(datetime(1983, 1, 1))

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        Window.bind(on_keyboard=self.events)
        self.manager_open = False
        self.file_manager = MDFileManager(
            exit_manager=self.exit_manager,
            select_path=self.select_path,
        )
        
    def build(self):
        self.screen = Builder.load_file("app.kv")
        return self.screen

    def get_PhoneNumberType(self):
        try:
            self.openModal()
            receive = self.requests_session.get(f'{API}/api/v1/lokati/PhoneNumberType/')
            if receive.status_code == 200 :

                r= list(receive.json())
           
            else :
                r= False

            return r
        except Exception as e :
            Logger.error(f'get_PhoneNumberType : {e} ')
            return False
        finally :
            pass

    def get_gata(self):
        Logger.info("App: Getting Data")
        if  self.store.exists('ports'):
            self.ports = self.store.get('ports')['data']
            print("ports",self.ports)

    def save_data(self):
        Logger.info("App: Saving Data")
        self.store.put('ports',data=self.ports)

    def open_add_country(self):
        if not self.Add_country:
            self.Add_country = Factory.Add_country()
        self.ModalAdd.clear_widgets()
        self.ModalAdd.add_widget(self.Add_country)

    def open_add_port(self):
        self.Add_port.init()
        self.ModalAdd.open()
        self.Add_port.ids.rv_constituant.update()
        # self.Add_port.ids.rv_data.update()
        
    def on_start(self):
        self.get_gata()
        Logger.info("App: Starting")
        if self.ModalAdd is None:
            self.ModalAdd = Factory.ModalAdd()
            if not self.Add_port:
                self.Add_port = Factory.Add_port()

            self.ModalAdd.add_widget(self.Add_port)
        Logger.info("App: Dialog Opened")
        # self.go_model_1()
        if not self.search_dropdown :
            self.search_dropdown = Search_dropdown(size_hint=(.75,.75))
            # self.search_dropdown.bind(selcted=self.select_country)
            countrys = list(self.ports.keys())

            self.selected_country = "null" if not len(countrys) else countrys[0]
        
            if self.selected_country == "null":
                ports = list(self.ports[self.selected_country])
                self.selected_port = ports[0] if len(ports) else "null"
             



                # ports= list(self.ports['Etats-Unis'].keys())
                # self.selcted =  ports[0] if len(ports) else ""

            else :
                self.selected_port = "null"
   
            # self.search_dropdown.set_list_md_icons()
            # self.search_dropdown.selcted = ports[0] if len(ports) else ""

            # print("selected",self.search_dropdown.selcted)


        countrys = list(self.ports.keys())
        country = countrys[0] if len(countrys) else "null"
        self.start_date=(datetime.now().strftime("%Y-%m-%d"))
        self.end_date=self.start_date
        self.select_port("country",country)

        print("start_date",self.start_date)
        # if   self.search_dropdown.datas == [] :
        #     phoneType = self.get_PhoneNumberType()
        #     if phoneType :
        #         default = [country for country in phoneType if country['region_code'] == "SN" ]
        #         self.selcted =  default[0] if default   else phoneType[0] 
        #         self.search_dropdown.datas = phoneType
        #         self.search_dropdown.set_list_md_icons()

    def select_country(self,instance,value):
        self.selcted = value
        # self.ids.country.text = value['display_name']
        self.search_dropdown.dismiss()      

    def go_model_1(self,country, port):
        Logger.info("App: Model 1")
        self.screen.clear_widgets()
        if self.model_1 is None:
            self.model_1 = Factory.Model_1()
        self.model_1.country = country
        self.model_1.port = port
        self.screen.add_widget(self.model_1)

    def go_menu(self):
        Logger.info("App: Menu")
        self.screen.clear_widgets()
        if self.countryScreen is None:
            self.countryScreen = Factory.CountryScreen()

        self.screen.add_widget(self.countryScreen)
    
    def show_country(self, country):
        Logger.info("App: Show Country")
        self.screen.clear_widgets()
        if self.portScreen is None:
            self.portScreen = Factory.PortScreen()
        self.portScreen.country = country
        self.portScreen.ids.rv_.country = country
        self.screen.add_widget(self.portScreen)
        self.portScreen.ids.rv_.update()
        # self.screen.add_widget(Factory.CountryCard())

    def calculate(self):
        Logger.info("App: Calculate")
        temps = np.arange(0, 10, 0.1)  # Temps de 0 à 10 secondes avec un pas de 0.1 seconde
        hauteur_eau = 3 * np.sin(2 * np.pi * 0.5 * temps) + 2 * np.sin(2 * np.pi * 2 * temps) + 0.5 * np.random.normal(size=len(temps))
        # Fonction de modèle avec une somme de cosinus basée sur les composantes de marée spécifiques
        DATA = self.ports["Etats-Unis"]["San Luis"]
        
        DATA_FILTERED = [v for k,v in DATA.items() if k in self.filterCOnstituant]
        print(DATA_FILTERED)
        return

        amplitudes = [ x[0] for x in DATA] 
        dephasages = [ x[1] for x in DATA]

        frequences = [ x[2] for x in DATA] 

        def modele_marée(t, Z, *parametres):
            resultat = Z
            for i in range(0, len(parametres), 3):
                Ac = parametres[i]
                Wc = 2*3.14*(parametres[i + 1])/360
                Lc = parametres[i + 2]
                resultat += Ac * np.cos(Wc * (t/3600) - Lc)
            return resultat

        # Estimation des paramètres du modèle à partir des données spécifiques
        parametres_initiaux = [Z] + [c for composante in zip(amplitudes, frequences, dephasages) for c in composante]
        # parametres_optimaux, _ = curve_fit(modele_marée, temps, hauteur_eau, p0=parametres_initiaux, maxfev=90000)

        # Générer des points temporels pour la prédiction
        temps_prediction = np.arange(0, 24, 0.1)

        # Faire la prédiction
        hauteur_eau_predite = modele_marée(temps_prediction, *parametres_initiaux)
        
        # Afficher les résultats
        # print(parametres_optimaux)

        

        Logger.info("calculate:parametres_initiaux : {}".format(parametres_initiaux)) 
        Logger.info("calculate:temps_prediction : {}".format(temps_prediction)) 
        Logger.info("calculate:hauteur_eau_predite : {}".format(hauteur_eau_predite)) 

        #self.model_1.ids.graph.prediction = list(hauteur_eau_predite)

        self.model_1.ids.graph.prediction = np.load("sanLuis.npy")
        

        print(len(hauteur_eau_predite))
        print(max(hauteur_eau_predite), min(hauteur_eau_predite),(max(hauteur_eau_predite) - min(hauteur_eau_predite)) )
        # liste_groupee = [(hauteur_eau_predite[i], hauteur_eau_predite[i + 1]) for i in range(0, len(hauteur_eau_predite), 2)]
        # print(len(liste_groupee))

        Logger.info("App: Calculate Done")

    def calculate2(self,constituant):
        Logger.info("App: Calculate")
        self.filterCOnstituant.append(constituant)
        
        self.calculate()
   
        '''def show_date_picker(self,sender):
        date_dialog = MDDatePicker(year=1983, month=4, day=12)
        date_dialog.bind(on_save=self.on_save, on_cancel=self.on_cancel)
        date_dialog.open()'''
   
    def show_date_picker(self, sender):
        date_dialog = MDDatePicker(mode="range")
        date_dialog.bind(on_save=self.on_save, on_cancel=self.on_cancel)
        date_dialog.open()

    def on_save(self, instance, value, date_range):
        '''
        Events called when the "OK" dialog box button is clicked.

        :type instance: <kivymd.uix.picker.MDDatePicker object>;

        :param value: selected date;
        :type value: <class 'datetime.date'>;

        :param date_range: list of 'datetime.date' objects in the selected range;
        :type date_range: <class 'list'>;
        '''
        if date_range:
            self.start_date = date_range[0].strftime("%Y-%m-%d")
            self.end_date = date_range[-1].strftime("%Y-%m-%d")
        else:
            self.start_date = value.strftime("%Y-%m-%d")
            self.end_date = self.start_date
        print("Range selected from", self.start_date, "to", self.end_date)

        self.prediction(self.screen.ids.home_screen.ids.graph)
        #111
        #print(instance, value, date_range)
        #self.start_date = value.strftime("%Y-%m-%d")

    def on_cancel(self, instance, value):
        '''Events called when the "CANCEL" dialog box button is clicked.'''

    def load_data(self,path):
        # csv_file_path = '0101-3112-2023(m).csv'
        data = pd.read_csv(path)
        snackbar = Snackbar(
            text="data loaded",
            snackbar_x="10dp",
            snackbar_y="10dp",
        )
        snackbar.open()
 
        self.train(data)
        # self.ports[self.Add_port.selected_country][self.Add_port.selected_port]['data'] = [ [a ,b ,c] for zip(self.tide_data['Verified (m)'].values) ]
        # self.Add_port.ids.rv_data.update()
        # # print(self.tide_data.head())

        # # self.Add_port.load_data()
        # # self.prediction2(self.screen.ids.home_screen.ids.graph)
        # self.Add_port.ids.rv_data.update()



    def prediction(self,graph):
        self.actual_predictions = []
        print("prediction",graph)
        # Define the constituents
        print("selected_port",self.selected_port)
        print(self.ports[self.selected_country][self.selected_port])

        constituents = [ (n,values[0],values[1]) for n,values in self.ports[self.selected_country][self.selected_port]['constituants'].items()]
       
        # Combine date and time into one datetime column (if necessary) and convert to seconds since epoch

      
        # Convert amplitudes from feet to meters and phases from degrees to radians
        amplitudes = np.array([amp for _, amp, _ in constituents]) 
        phases = np.array([phase for _, _, phase in constituents])
        # speeds = np.array([speed for _,  _, _, speed in constituents]) 

        speeds = np.array([speed for  _,speed in self.speeds.items()]) * np.pi / 180 / 3600
        msl = self.ports[self.selected_country][self.selected_port]["msl"]

        # Prediction function
  
        # Generate timestamps for predictions
        start_datetime = datetime.strptime(self.start_date, "%Y-%m-%d")
        timestamps = np.array([(start_datetime + timedelta(hours=i)).timestamp() for i in range(24)])
        times_since_epoch = (timestamps - self.epoch.timestamp())
        predicted_optimized = self.predict_tide(msl,times_since_epoch, amplitudes, phases, speeds)
        
        # graph.reset_graph()
        graph.update_graph(predicted_optimized)
        graph.predicted_optimized_plot.points = [(i,( p)) for i,p in enumerate(predicted_optimized)]

        def find_high_low_tides(times, predicted_heights):
            high_tides = []
            low_tides = []
            self.actual_predictions.append((times[0],predicted_heights[0],"----"))
            for i in range(1, len(predicted_heights) - 1):
                if predicted_heights[i] > predicted_heights[i - 1] and predicted_heights[i] > predicted_heights[i + 1]:
                    # high_tides.append((times[i], predicted_heights[i]))
                    self.actual_predictions.append((times[i],predicted_heights[i],"Haute"))
                elif predicted_heights[i] < predicted_heights[i - 1] and predicted_heights[i] < predicted_heights[i + 1]:
                    # low_tides.append((times[i], predicted_heights[i]))
                    self.actual_predictions.append((times[i],predicted_heights[i],"Basse"))
                else:
                    self.actual_predictions.append((times[i],predicted_heights[i],"----"))
                    
            self.actual_predictions.append((times[-1],predicted_heights[-1],"----"))

            

        # Find high and low tides
        find_high_low_tides(timestamps, predicted_optimized)

        #add desigation
        # high_tides = [(t, h, 'Haute') for t, h in high_tides]
        # low_tides = [(t, h, 'Basse') for t, h in low_tides]

        # join high and low tides
        # high_tides = high_tides + low_tides

        # Sort high tides by time
        # high_tides.sort(key=lambda x: x[0])

        print("High Tides:")
        index=1
        self.screen.ids.home_screen.ids.maree_list.clear_widgets()
        for time, height ,designation in self.actual_predictions:
            # get only hours and minutes
            time = datetime.fromtimestamp(time).strftime("%H:%M")
            print(f"{time}: {height:.4f} meters")
            self.screen.ids.home_screen.ids.maree_list.add_widget(Factory.Single_maree(designation=designation ,index=index,heure=str(time), hauteur=f"{height:.4f} m"))
            index+=1

        # for   c,a,p,s  in zip(constutuant_names,optimized_amplitudes, optimized_phases, speeds):

        #     self.ports[self.selected_country][self.selected_port]['constituants'][c] = [a,p,s]

        # Predict tide heights for these times using initial and optimized parameters
        # predicted_initial = predict_tide(times_since_epoch, amplitudes, phases, speeds)
        # for plot in graph.graph.plots:
        #     graph.graph.remove_plot(plot)

        # graph.update_graph(predicted_initial)

        # predicted_initial_plot = MeshLinePlot(color=[0, 0, 1, 1])
        # predicted_initial_plot.points = [(i,( p)) for i,p in enumerate(predicted_initial)]

        # graph.add_plot(predicted_initial_plot)
        # # graph.graph.add_plot(predicted_initial_plot)
        # graph.graph.add_plot(predicted_optimized_plot)

        # graph.add_plot(noaa_verified_heights_plot)
        # graph.update_graph(noaa_verified_heights)
        # plt.figure(figsize=(12, 6))
        # plt.plot(timestamps, predicted_initial, label='Initial Prediction', marker='o')
        # plt.plot(timestamps, predicted_optimized, label='Optimized Prediction', marker='x')
        # plt.plot(noaa_timestamps, noaa_verified_heights, label='NOAA Verified (New Data)', marker='s', linestyle='--', color='red')

        # plt.xlabel('Hour of the Day on 10/01/2024')
        # plt.ylabel('Tide Height (meters)')
        # plt.title('24-Hour Tide Height Prediction Before and After Optimization with New NOAA Verified Data')
        # plt.xticks(timestamps, [f"{i}:00" for i in range(24)], rotation=45)
        # plt.legend()
        # plt.tight_layout() # Adjust layout to prevent clipping of tick labels
        # plt.show()<n



    # Prediction function
    def predict_tide(self,msl,times, amplitudes, phases, speeds):
        """Predict tide heights for a given set of times using the provided amplitudes, phases, and speeds."""
            
        # msl_2023 = tide_data['Verified (m)'].mean()
        tide_height = np.sum(amplitudes * np.cos(speeds * times[:, None] + phases), axis=1)
        return msl + tide_height


    def train(self,data):
        """Train the tide model using the provided tide data."""
        Logger.info("App: Train")

        # Define the constituents
        constituents = [ (n,values[0],values[1]) for n,values in self.ports[self.Add_port.selected_country][self.Add_port.selected_port]['constituants'].items()]
        constutuant_names = [n for n,values in self.ports[self.Add_port.selected_country][self.Add_port.selected_port]['constituants'].items()]
  
        # Load the tide data from CSV
        data['Datetime'] = pd.to_datetime(data['Date'] + ' ' + data['Time (GMT)'])
        data['Seconds since epoch'] = (data['Datetime'] - self.epoch).dt.total_seconds()

        # Extract observed heights and times
        observed_heights = data['Verified (m)'].values
        times = data['Seconds since epoch'].values



        # Convert amplitudes from feet to meters and phases from degrees to radians
        amplitudes = np.array([amp for _, amp, _ in constituents]) * 0.3048
        phases = np.array([phase for _, _, phase in constituents]) * np.pi / 180
        speeds = np.array([speed for  _,speed in self.speeds.items()]) * np.pi / 180 / 3600
        msl= data['Verified (m)'].mean()

        # Prediction function

        # Loss function
        def loss_fn(params, times, observed_heights):
            """Calculate the mean squared error between the observed and predicted tide heights."""
            num_constituents = len(speeds)
            amplitudes = params[:num_constituents]
            phases = params[num_constituents:]
            predictions = self.predict_tide(msl,times, amplitudes, phases, speeds)
            return np.mean((predictions - observed_heights) ** 2)

        # Initial parameters
        initial_params = np.concatenate([amplitudes, phases])

        # Marquer le début de l'optimisation
        start_time = datetime.now()

        # Run the optimizer
        res = minimize(
            fun=loss_fn,
            x0=initial_params,
            args=(times, observed_heights),
            method='L-BFGS-B',
            options={'maxfun': 50000, 'maxiter': 50000}
        )

        # Marquer la fin de l'optimisation
        end_time = datetime.now()

        # Calculer la durée
        duration = end_time - start_time

        if res.success:
            optimized_params = res.x
            optimized_amplitudes = optimized_params[:len(speeds)]
            optimized_phases = optimized_params[len(speeds):]
            print(len(speeds))
            print("Optimization successful.")
            print(f"Optimization time: {duration}")

            # Affichage des constituants optimisés
            print("Optimized Constituents:")
            for i, constituent in enumerate(constituents):
                name = constituent[0]
                print(f"{name}: Amplitude (meters) = {optimized_amplitudes[i]:.4f}, Phase (degrees) = {optimized_phases[i]:.4f}")
        else:
            print("Optimization failed.")
            print(f"Optimization failed with message: {res.message}")



        # Load the new NOAA data
        # noaa_data['Datetime'] = pd.to_datetime(noaa_data['Date'] + ' ' + noaa_data['Time (GMT)'])

        # Generate timestamps for predictions
        start_datetime = datetime.strptime(data['Date'][0], "%Y/%m/%d")
        # 2023-01-01
        start_datetime=datetime(2023, 1, 1)
        data = data[:24]

        timestamps = np.array([(start_datetime + timedelta(hours=i)).timestamp() for i in range(24)])
        print("timestamps",start_datetime)
        print("times",data)
        times_since_epoch = (timestamps - self.epoch.timestamp())


        predicted_optimized = self.predict_tide(msl,times_since_epoch, optimized_amplitudes, optimized_phases, speeds)


        def calculate_rmse(observed, predicted):
            return np.sqrt(mean_squared_error(observed, predicted))
        

        noaa_verified_heights = data['Verified (m)'][:24].values

        rmse = calculate_rmse(noaa_verified_heights, predicted_optimized)

        print(f"RMSE: {rmse}")
        print('Optimized amplitude ' )
        # noaa_verified_heights_plot = MeshLinePlot(color=[1, 0, 0, 1])
        # noaa_verified_heights_plot.points = [(i,( p)) for i,p in enumerate(noaa_verified_heights)]
        # # self.screen.ids.home_screen.ids.graph.graph.add_plot(noaa_verified_heights_plot)
        # self.prediction(self.screen.ids.home_screen.ids.graph)

        self.ports[self.Add_port.selected_country][self.Add_port.selected_port]["RMSE"] = rmse
        self.ports[self.Add_port.selected_country][self.Add_port.selected_port]["msl"] = msl
        constituants = {}
        for   c,a,p  in zip(constutuant_names,optimized_amplitudes, optimized_phases):

            constituants[c] = [a,p]
        self.ports[self.Add_port.selected_country][self.Add_port.selected_port]['constituants'] = constituants
        self.Add_port.ids.rv_constituant.update()
        Logger.info(f"port: {self.ports}")
        self.save_data()
        Snackbar(
            text="Model trained",
            snackbar_x="10dp",
            snackbar_y="10dp",
        ).open()


    


    def file_manager_open(self):
        "open the file manager"
        self.file_manager.show(os.path.abspath(os.getcwd()))  # output manager to the screen
        self.manager_open = True

    def select_path(self, path):
        '''It will be called when you click on the file name
        or the catalog selection button.

        :type path: str;
        :param path: path to the selected directory or file;
        '''
        print(path)
        self.load_data(path)
        self.exit_manager()
        
    def exit_manager(self, *args):
        '''Called when the user reaches the root of the directory tree.'''

        self.manager_open = False
        self.file_manager.close()

    def events(self, instance, keyboard, keycode, text, modifiers):
        '''Called when buttons are pressed on the mobile device.'''

        if keyboard in (1001, 27):
            if self.manager_open:
                self.file_manager.back()
    
        return True
    
    def show_add_country_dialog(self):
        "show the dialog country"
        if not self.dialog_country:
            self.dialog_country = MDDialog(
                title="Nouveau:",
                type="custom",
                content_cls=Add_country_dialog(),
                buttons=[
                    MDFlatButton(
                        text="CANCEL", text_color=self.theme_cls.primary_color , on_release=self.dismiss_dialog_country
                    ),
                    MDFlatButton(
                        text="Save", text_color=self.theme_cls.primary_color ,on_release=self.new_country,
                    ),
                ],
            )
        self.dialog_country.open()

    def show_add_port_dialog(self):
        "show the dialog port"
        if not self.dialog_port:
            self.dialog_port = MDDialog(
                title="Nouveau:",
                type="custom",
                content_cls=Add_port_dialog(),
                buttons=[
                    MDFlatButton(
                        text="CANCEL", text_color=self.theme_cls.primary_color , on_release=self.dismiss_dialog_port
                    ),
                    MDFlatButton(
                        text="OK", text_color=self.theme_cls.primary_color , on_release= self.new_port ,

                    ),
                ],
            )
        self.dialog_port.open()

    def dismiss_dialog_port(self, *args):
        "dismiss the dialog port"
        self.dialog_port.dismiss()

    def dismiss_dialog_country(self, *args):
        "dismiss the dialog country"
        self.dialog_country.dismiss()

    def new_country(self, *args):
        "add a new country to the ports"
        print("new_country")
        country = self.dialog_country.content_cls.ids.country.text

        lowered_country = [c.lower() for c in self.ports]

        if not country or country.lower() in lowered_country:
            Snackbar(text="Country already exists").open()
            return
        self.ports[country] = {}


        self.Add_port.select_port( "country", country)
        self.dialog_country.dismiss()
        
    def new_port(self, *args):
        "add a new port to the selected country"
        print("new_port")
        print(self.dialog_port.content_cls.ids.port.text)
        port = self.dialog_port.content_cls.ids.port.text
        
        lowered_ports = [c.lower() for c in self.ports[self.Add_port.selected_country]]

        if not port or port in lowered_ports :
            Snackbar(text="Port already exists").open()
            return
        self.ports[self.Add_port.selected_country][port] = {
            "data": {},
            "constituants": {constituant: [0, 0, 0] for constituant in self.list_training_constituants},
            "prediction": [],
            "RMSE": 0,
            "msl":0
        }
        self.Add_port.select_port("port",port)
        self.dialog_port.dismiss()
    
    def select_port(self, search_mode, value):
        "change the selected port and update the graph"
        if search_mode == "country":
            self.selected_country = value
            ports = list(self.ports[self.selected_country])
            self.selected_port = ports[0] if len(ports) else ""


        elif search_mode == "port":

            self.selected_port = value

        self.prediction(self.screen.ids.home_screen.ids.graph)
    def change_country(self, country):
        "change the selected country and update the selected port to the first port of the country"
        self.selected_country = country
        ports = list(self.ports[self.selected_country])
        self.selected_port = ports[0] if len(ports) else ""


    def screen_shoot(self):
        "take a screenshot of the graph and save it as screenshot.png"
        self.screen.ids.home_screen.ids.graph.export_to_png("screenshot.png")
        Snackbar(text="Screenshot saved").open()
    
    def export_data(self):
        "use json to save data ports"
        # with open("sample.json", "w") as outfile: 
        #     json.dump(self.ports, outfile)
        self.screen.ids.home_screen.ids.graph.export_to_png("screenshot.png")
        page = Factory.Export_DAY(date=self.start_date,size_hint=(.50,.80))
        #copy the graph and the list of maree
        # from copy import copy
        # page.ids.graph.add_widget(copy(self.screen.ids.home_screen.ids.graph))
        # page.ids.maree_list.add_widget(self.screen.ids.home_screen.ids.maree_list)

        # export to png
   
        # page.export_to_png("screenshot.png")
        # self.screen.ids.home_screen.ids.maree_list.export_to_png("screenshot.png")
    
        # Snackbar(text="Data exported").open()

        index=1
        page.ids.maree_list.clear_widgets()
        for time, height ,designation in self.actual_predictions:
            # get only hours and minutes
            time = datetime.fromtimestamp(time).strftime("%H:%M")
            print(f"{time}: {height:.4f} meters")
            page.ids.maree_list.add_widget(Factory.Single_maree(designation=designation ,index=index,heure=str(time), hauteur=f"{height:.4f} m"))
            index+=1
        page.open()
        clock = Clock.schedule_once(lambda dt: page.export_to_png("export.png"), 1)
        # page.export_to_png("export.png")


    def save_data(self):
        "use json to save data ports"
        self.store.put('ports',data=self.ports)

    def save_model(self):
        "use pickle to save model"
        pass

    def otrain(self,):
        "Old train function"
        graph = self.screen.ids.home_screen.ids.graph
        print("prediction",graph)
        # Define the constituents
        constituents = [ (n,values[0],values[1],values[2]) for n,values in self.ports[self.selected_country][self.selected_port]['constituants'].items()]
        constutuant_names = [n for n,values in self.ports[self.selected_country][self.selected_port]['constituants'].items()]

        # Load the tide data from CSV
        csv_file_path = '0101-3112-2023(m).csv'
        tide_data = pd.read_csv(csv_file_path)

        # Combine date and time into one datetime column (if necessary) and convert to seconds since epoch
        epoch = datetime(1983, 1, 1)
        tide_data['Datetime'] = pd.to_datetime(tide_data['Date'] + ' ' + tide_data['Time (GMT)'])
        tide_data['Seconds since epoch'] = (tide_data['Datetime'] - epoch).dt.total_seconds()

        # Extract observed heights and times
        observed_heights = tide_data['Verified (m)'].values
        times = tide_data['Seconds since epoch'].values

        # Convert amplitudes from feet to meters and phases from degrees to radians
        amplitudes = np.array([amp for _, amp, _, _ in constituents]) * 0.3048
        phases = np.array([phase for _, _, phase, _ in constituents]) * np.pi / 180
        speeds = np.array([speed for _, _, _, speed in constituents]) * np.pi / 180 / 3600

        # Prediction function
        def predict_tide(times, amplitudes, phases, speeds):
            tide_height = np.sum(amplitudes * np.cos(speeds * times[:, None] + phases), axis=1)
            return tide_height

        # Loss function
        def loss_fn(params, times, observed_heights):
            num_constituents = len(speeds)
            amplitudes = params[:num_constituents]
            phases = params[num_constituents:]
            predictions = predict_tide(times, amplitudes, phases, speeds)
            return np.mean((predictions - observed_heights) ** 2)

        # Initial parameters
        initial_params = np.concatenate([amplitudes, phases])

        # Marquer le début de l'optimisation
        start_time = datetime.now()

        # Run the optimizer
        res = minimize(
            fun=loss_fn,
            x0=initial_params,
            args=(times, observed_heights),
            method='L-BFGS-B',
            options={'maxfun': 50000, 'maxiter': 50000}
        )

        # Marquer la fin de l'optimisation
        end_time = datetime.now()

        # Calculer la durée
        duration = end_time - start_time

        if res.success:
            optimized_params = res.x
            optimized_amplitudes = optimized_params[:len(speeds)]
            optimized_phases = optimized_params[len(speeds):]
            print("Optimization successful.")
            print(f"Optimization time: {duration}")

            # Affichage des constituants optimisés
            print("Optimized Constituents:")
            for i, constituent in enumerate(constituents):
                name = constituent[0]
                print(f"{name}: Amplitude (meters) = {optimized_amplitudes[i]:.4f}, Phase (degrees) = {optimized_phases[i]:.4f}")
        else:
            print("Optimization failed.")
            print(f"Optimization failed with message: {res.message}")

        # Load the new NOAA data
        noaa_data_path = '10-01-2023.csv'
        noaa_data = pd.read_csv(noaa_data_path)
        noaa_data['Datetime'] = pd.to_datetime(noaa_data['Date'] + ' ' + noaa_data['Time (GMT)'])
        noaa_timestamps = noaa_data['Datetime'].apply(lambda x: x.timestamp()).values
        noaa_verified_heights = noaa_data['Verified (m)'].values

        # Generate timestamps for predictions
        start_datetime = datetime(2024, 1, 10)
        timestamps = np.array([(start_datetime + timedelta(hours=i)).timestamp() for i in range(24)])
        times_since_epoch = (timestamps - epoch.timestamp())


        predicted_optimized = predict_tide(times_since_epoch, optimized_amplitudes, optimized_phases, speeds)

    

        for   c,a,p,s  in zip(constutuant_names,optimized_amplitudes, optimized_phases, speeds):

            self.ports[self.selected_country][self.selected_port]['constituants'][c] = [a,p,s]

        def calculate_rmse(observed, predicted):
            return np.sqrt(mean_squared_error(observed, predicted))

        rmse = calculate_rmse(noaa_verified_heights, predicted_optimized)
        print(f"RMSE: {rmse}")

        print('Optimized amplitude ' )

 

        # predicted_optimized_plot = MeshLinePlot(color=[.5, .5, 0, 1])
        # predicted_optimized_plot.points = [(i,( p)) for i,p in enumerate(predicted_optimized)]

        # graph.update_graph(predicted_optimized)
        # graph.graph.add_plot(predicted_optimized_plot)



        # noaa_verified_heights_plot = MeshLinePlot(color=[1, 0, 0, 1])
        # noaa_verified_heights_plot.points = [(i,( p)) for i,p in enumerate(noaa_verified_heights)]

        # graph.update_graph(noaa_verified_heights)
        # graph.graph.add_plot(noaa_verified_heights_plot)



        # plt.figure(figsize=(12, 6))
        # plt.plot(timestamps, predicted_initial, label='Initial Prediction', marker='o')
        # plt.plot(timestamps, predicted_optimized, label='Optimized Prediction', marker='x')
        # plt.plot(noaa_timestamps, noaa_verified_heights, label='NOAA Verified (New Data)', marker='s', linestyle='--', color='red')

        # plt.xlabel('Hour of the Day on 10/01/2024')
        # plt.ylabel('Tide Height (meters)')
        # plt.title('24-Hour Tide Height Prediction Before and After Optimization with New NOAA Verified Data')
        # plt.xticks(timestamps, [f"{i}:00" for i in range(24)], rotation=45)
        # plt.legend()
        # plt.tight_layout() # Adjust layout to prevent clipping of tick labels
        # plt.show()
    def change_day(self, day):
        "change day for the prediction"
    
        # previous_date = datetime.strptime(self.start_date, "%Y-%m-%d") - timedelta(days=day)
        next_date = datetime.strptime(self.start_date, "%Y-%m-%d") + timedelta(days=day)
        self.start_date = next_date.strftime("%Y-%m-%d") 

        self.prediction(self.screen.ids.home_screen.ids.graph)

    # add  if day == -1  or day == 1 add 1 or -1 to the start_date

 
    


MainApp().run()


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


class Add_port(ModalView):
    pass

class Search_dropdown(ModalView):
    datas = ListProperty([])
    selcted =  StringProperty("")

    def __init__(self, **kwargs):
        super(Search_dropdown, self).__init__(**kwargs)
        self.app = MDApp.get_running_app()


    def set_list_md_icons(self , text="", search=False):

        '''Builds a list of icons for the screen MDIcons.'''
    
        self.datas =  self.app.ports['Etats-Unis']

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
        self.selcted = value
        self.dismiss()


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
        constitauns = self.app.ports['Etats-Unis']['San Luis']

        self.data = [{'text': str(c) ,'tooltip_text':str(v[3])  } for c , v in constitauns.items()]

    def update(self):
        if  not self.country :
            return
        # ports = self.app.ports[self.country]
        constitauns = self.app.ports['Etats-Unis']['San Luis']


        self.data = [{'text': str(c) ,'tooltip_text':str(v[3])  } for c , v in constitauns.items()]

    # def search(self , text="",):
    #     if  not self.country :
    #         return
        
    #     self.data = []
    #     results = []
    #     ports = self.app.ports[self.country]
    #     for c in ports:
    #         if text.lower() in c.lower():
    #             results.append(c)
 
    #     self.data = [{'country':self.country,'text': str(country)  } for country in results]




class GraphModel_1(MDRelativeLayout):
    grid_nombre = NumericProperty(24)
    prediction = ListProperty([])

    def __init__(self, **kwargs):
        super(GraphModel_1, self).__init__(**kwargs)
        # self.prediction = np.arange(0, 24, 0.1)
        self.prediction= np.load("sanLuis.npy")

        self.graph = Graph(xlabel="", ylabel="y", x_ticks_major=1, y_ticks_major=1,
                           y_grid_label=True, x_grid_label=True, x_grid=False, y_grid=True,
                           xmin=0, xmax=100, ymin=-1, ymax=1, draw_border=False,label_options={'color': (0, 0, 0, 1),},)
        # self.graph.size = (1200, 400)
        # self.graph.pos = self.center
        # self.graph.pos = self.pos
        gap = self.width / len(self.prediction)

        self.plot = MeshLinePlot(color=[1, 0, 0, 1])
        # self.plot.points = [(i, self.height /2 + ( p)) for i,p in enumerate(self.prediction)]

        # self.plot2 = MeshLinePlot(color=[0, 0, 0, 1])
        # self.plot2.points = [(x, cos(x / 10.)) for x in range(0, 101)]

        self.add_widget(self.graph)
        self.graph.add_plot(self.plot)
        self.bind(pos=self.updateDisplay)
        self.bind(size=self.updateDisplay)

        self.bind(prediction=self.updateDisplay)

    def updateDisplay(self, *args):
        

        self.graph.size = (self.width, self.height)

        # if self.prediction:
        
        #     # self.plot.points = [(x, sin(x / 10.)) for x in range(0, 101)]

        #     # print(self.prediction)
            
        #     self.plot.points = [(i,  p) for i,p in enumerate(self.prediction)]

 
            
    def update_graph(self, plot):

        self.graph.xmax = len(plot)
        self.graph.x_ticks_major = len(plot) / 10
        self.graph.ymin = int(min(plot)) -1
        self.graph.ymax = int(max(plot)) + 1





        



class MainApp(MDApp):
    
    start_date = StringProperty("2023-10-01")#111
    end_date = StringProperty("2023-10-01")#111
    model_1 = ObjectProperty(None)
    portScreen = ObjectProperty(None)
    countryScreen = ObjectProperty(None)
    filterCOnstituant = ListProperty([])
    search_dropdown = ObjectProperty(None,allownone=True)
    selcted =StringProperty("")   
    date_mode= StringProperty("from") 
    from_date = StringProperty("2023-10-01")  
    tide_data = ObjectProperty(None)
    store = ObjectProperty(JsonStore('data.json'))
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
    

    ports = DictProperty({"France": {},
                        "Etats-Unis": {
                            "Los Angeles" : [],
                            "New York" : [],
                            "San Francisco" : [],
                            "San Luis" : {
                            "constituants": {
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
        if  self.cache.exists('user'):
            self.ports = self.store.get('ports')


    def save_data(self):
        Logger.info("App: Saving Data")
        self.store.put('ports',data=self.ports)

    def on_start(self):
        Logger.info("App: Starting")

        Logger.info("App: Dialog Opened")
        # self.go_model_1()
        if not self.search_dropdown :
            self.search_dropdown = Search_dropdown(size_hint=(.75,.75))
            self.search_dropdown.bind(selcted=self.select_country)

            ports= list(self.ports['Etats-Unis'].keys())
            self.selcted =  ports[0] if len(ports) else ""

    
            self.search_dropdown.set_list_md_icons()
            self.search_dropdown.selcted = ports[0] if len(ports) else ""

            print("selected",self.search_dropdown.selcted)

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
    #111
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

        #111
        #print(instance, value, date_range)
        #self.from_date = value.strftime("%Y-%m-%d")

    def on_cancel(self, instance, value):
        '''Events called when the "CANCEL" dialog box button is clicked.'''

    def load_data(self,path):
        # csv_file_path = '0101-3112-2023(m).csv'
        self.tide_data = pd.read_csv(path)
        snackbar = Snackbar(
            text="data loaded",
            snackbar_x="10dp",
            snackbar_y="10dp",
        )
        snackbar.open()
        self.predcition2(self.screen.ids.home_screen.ids.graph)


    def train(self):
        Logger.info("App: Train")
        
        constituents = [
            ('M2', 1.8, 190.4, 28.984104),
            ('S2', 0.45, 196.3, 30.0),
            ('N2', 0.4, 164.1, 28.43973),
            ('K1', 1.26, 219.9, 15.041069),
            ('O1', 0.77, 204.1, 13.943035),
            ('P1', 0.39, 215.0, 14.958931),
            ('M4', 0.0, 259.1, 57.96821)
        ]

    def predcition2(self,graph):

        print("prediction",graph)
        # Define the constituents



        if self.tide_data is not None:

            # Combine date and time into one datetime column (if necessary) and convert to seconds since epoch
            epoch = datetime(1983, 1, 1)
            self.tide_data['Datetime'] = pd.to_datetime(self.tide_data['Date'] + ' ' + self.tide_data['Time (GMT)'])
            self.tide_data['Seconds since epoch'] = (self.tide_data['Datetime'] - epoch).dt.total_seconds()

            # Extract observed heights and times
            observed_heights = self.tide_data['Verified (m)'].values
            times = self.tide_data['Seconds since epoch'].values

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


       

        # Load the new NOAA data
        noaa_data_path = '10-01-2023.csv'
        noaa_data = self.tide_data

        noaa_data['Date'] = pd.to_datetime(noaa_data['Date'])
        noaa_data = noaa_data.set_index('Date')
        
        noaa_data = noaa_data.loc[self.from_date:self.from_date]
        print(noaa_data)
        
        # noaa_data['Datetime'] = pd.to_datetime(noaa_data['Date'] + ' ' + noaa_data['Time (GMT)'])
        noaa_timestamps = noaa_data['Datetime'].apply(lambda x: x.timestamp()).values
        noaa_verified_heights = noaa_data['Verified (m)'].values

        # Generate timestamps for predictions
        start_datetime = datetime(2024, 1, 10)
        timestamps = np.array([(start_datetime + timedelta(hours=i)).timestamp() for i in range(24)])
        times_since_epoch = (timestamps - epoch.timestamp())

        # Predict tide heights for these times using initial and optimized parameters
        predicted_initial = predict_tide(times_since_epoch, amplitudes, phases, speeds)
     

        def calculate_rmse(observed, predicted):
            return np.sqrt(mean_squared_error(observed, predicted))



        print('Optimized amplitude ' )
        for plot in graph.graph.plots:
            graph.graph.remove_plot(plot)

        predicted_initial_plot = MeshLinePlot(color=[0, 0, 1, 1])
        predicted_initial_plot.points = [(i,( p)) for i,p in enumerate(predicted_initial)]

        graph.update_graph(predicted_initial)
        graph.graph.add_plot(predicted_initial_plot)




        noaa_verified_heights_plot = MeshLinePlot(color=[1, 0, 0, 1])
        noaa_verified_heights_plot.points = [(i,( p)) for i,p in enumerate(noaa_verified_heights)]

        graph.update_graph(noaa_verified_heights)
        graph.graph.add_plot(noaa_verified_heights_plot)

    def predcition(self,graph):

        print("prediction",graph)
        # Define the constituents
        constituents = [
            ('M2', 1.8, 190.4, 28.984104),
            ('S2', 0.45, 196.3, 30.0),
            ('N2', 0.4, 164.1, 28.43973),
            ('K1', 1.26, 219.9, 15.041069),
            ('O1', 0.77, 204.1, 13.943035),
            ('P1', 0.39, 215.0, 14.958931),
            ('M4', 0.0, 259.1, 57.96821)
        ]

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

        # Predict tide heights for these times using initial and optimized parameters
        predicted_initial = predict_tide(times_since_epoch, amplitudes, phases, speeds)
        predicted_optimized = predict_tide(times_since_epoch, optimized_amplitudes, optimized_phases, speeds)

        def calculate_rmse(observed, predicted):
            return np.sqrt(mean_squared_error(observed, predicted))

        rmse = calculate_rmse(noaa_verified_heights, predicted_optimized)
        print(f"RMSE: {rmse}")

        print('Optimized amplitude ' )

        predicted_initial_plot = MeshLinePlot(color=[0, 0, 1, 1])
        predicted_initial_plot.points = [(i,( p)) for i,p in enumerate(predicted_initial)]

        graph.update_graph(predicted_initial)
        graph.graph.add_plot(predicted_initial_plot)

        predicted_optimized_plot = MeshLinePlot(color=[.5, .5, 0, 1])
        predicted_optimized_plot.points = [(i,( p)) for i,p in enumerate(predicted_optimized)]

        graph.update_graph(predicted_optimized)
        graph.graph.add_plot(predicted_optimized_plot)



        noaa_verified_heights_plot = MeshLinePlot(color=[1, 0, 0, 1])
        noaa_verified_heights_plot.points = [(i,( p)) for i,p in enumerate(noaa_verified_heights)]

        graph.update_graph(noaa_verified_heights)
        graph.graph.add_plot(noaa_verified_heights_plot)



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

    def file_manager_open(self):
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
    
    

MainApp().run()

'''


def update_graph(self, plot_data):
    days = (self.end_date - self.start_date).days + 1 
    hours = days * 24  # تعداد ساعت‌های انتخاب شده
    
    self.graph.xmax = hours
    self.graph.x_ticks_major = hours / days  # نمایش هر روز به عنوان یک میانگین بزرگ
    self.graph.ymax = max(plot_data)
    self.graph.ymin = min(plot_data)
    
    self.plot.points = [(i, p) for i, p in enumerate(plot_data)]
    self.graph.add_plot(self.plot)



def calculate(self):
 
    filtered_data = data[(data['date'] >= self.start_date) & (data['date'] <= self.end_date)]

def on_save(self, instance, value, date_range):
    if date_range:
        self.start_date = datetime.strptime(date_range[0], "%Y-%m-%d")
        self.end_date = datetime.strptime(date_range[1], "%Y-%m-%d")
    else:
        self.start_date = self.end_date = datetime.strptime(value, "%Y-%m-%d")
    
    # پس از ذخیره تاریخ، نمودار را به‌روزرسانی کنید.
    self.calculate()
'''
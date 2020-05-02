# -*- coding: utf-8 -*-

import flask
import dash
import dash_table
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Output, Input, State

#from random import randint

import os
import pathlib
import re
import json
from datetime import datetime
import string
import matplotlib.colors as mcolors
import plotly.graph_objs as go
import plotly.express as px
import pandas as pd
import numpy as np
import nltk
from dateutil import relativedelta
from wordcloud import WordCloud, STOPWORDS
from sklearn.feature_extraction.text import CountVectorizer
from operator import add

# Setup the app
# Make sure not to change this file name or the variable names below,
# the template is configured to execute 'server' on 'app.py'
#server = flask.Flask(__name__)
#server.secret_key = os.environ.get('secret_key', str(randint(0, 1000000)))
#app = dash.Dash(__name__, server=server)

WNL = nltk.WordNetLemmatizer()

f=open('source/firstpost.json')
occ_dic=json.load(f)
f.close()

def wc_yearlevel(lem,year):
	try:
		for month in occ_dic[lem][year]:
			try:
				result=list(map(add,result,occ_dic[lem][year][month]['Word_Count']))
			except:
				result=occ_dic[lem][year][month]['Word_Count']
	except:
		result=[0,0,0,0,0,0,0,0,0,0,0,0,0]
	return result

ngram_df = pd.read_csv("source/ngram_counts_data.csv", index_col=0)

DATA_PATH = pathlib.Path(__file__).parent.resolve()
EXTERNAL_STYLESHEETS = ["https://codepen.io/chriddyp/pen/bWLwgP.css"]
FILENAME = "source/firstpost.csv"
PLOTLY_LOGO = "https://images.plot.ly/logo/new-branding/plotly-logomark.png"
GLOBAL_DF = pd.read_csv(DATA_PATH.joinpath(FILENAME), header=0)
"""
We are casting the whole column to datetime to make life easier in the rest of the code.
It isn't a terribly expensive operation so for the sake of tidyness we went this way.
"""
GLOBAL_DF["Date"] = pd.to_datetime(
    GLOBAL_DF["Date"], format="%d/%m/%y"
)

"""
In order to make the graphs more useful we decided to prevent some words from being included
"""
ADDITIONAL_STOPWORDS = [
    "XXXX",
    "XX",
    "xx",
    "xxxx",
    "n't"
]
for stopword in ADDITIONAL_STOPWORDS:
    STOPWORDS.add(stopword)


"""
#  Somewhat helpful functions
"""


def sample_data(dataframe, float_percent):
    """
    Returns a subset of the provided dataframe.
    The sampling is evenly distributed and reproducible
    """
    print("making a local_df data sample with float_percent: %s" % (float_percent))
    return dataframe.sample(frac=float_percent, random_state=1)


def get_complaint_count_by_company(dataframe):
    """ Helper function to get complaint counts for unique themes """
    company_counts = dataframe["Theme"].value_counts()
    # we filter out all themes with less than 11 complaints for now
    company_counts = company_counts[company_counts > 10]
    values = company_counts.keys().tolist()
    counts = company_counts.tolist()
    check=0
    for count in counts:
    	check+=count
    counts.append(check)
    values.append("all")
    return values, counts

def get_count_by_ngram(dataframe):
    """ Helper function to get complaint counts for unique themes """
    company_counts = dataframe["ngram"].value_counts()
    # we filter out all themes with less than 11 complaints for now
    company_counts = company_counts[company_counts > 10]
    values = company_counts.keys().tolist()
    counts = company_counts.tolist()
    return values, counts


def calculate_themes_sample_data(dataframe, sample_size, time_values):
    """ TODO """
    print(
        "making themes_sample_data with sample_size count: %s and time_values: %s"
        % (sample_size, time_values)
    )
    if time_values is not None:
        min_date = time_values[0]
        max_date = time_values[1]
        dataframe = dataframe[
            (dataframe["Date"] >= min_date)
            & (dataframe["Date"] <= max_date)
        ]
    company_counts = dataframe["Theme"].value_counts()
    company_counts_sample = company_counts[:sample_size]
    values_sample = company_counts_sample.keys().tolist()
    counts_sample = company_counts_sample.tolist()

    return values_sample, counts_sample

def calculate_word_data(theme, word):
	if word != "data":
		lem=word
	else:
		lem="datum"
	year_list=["2010","2011","2012","2013","2014","2015","2016","2017","2018","2019","2020"]
	count_list=[]
	themelist=['all',
	'aadhaar_based_subsidies',
	'aadhaar_based_schemes',
	'digital_stack',
	'enrolment_process',
	'e-governance',
	'financial_inclusion',
	'macroeconomic_policy',
	'data_security',
	'aadhaar_political_debates',
	'judiciary_right_to_privacy',
	'money_laundering',
	'crime']
	for i in range(0,13):
		if theme==themelist[i]:
			rant=i
			break
	for year in year_list:
		result=wc_yearlevel(lem,year)
		count_list.append(result[i])

	#print(count_list)

	return year_list, count_list

def make_local_df(selected_themes, time_values, n_selection):
    """ TODO """
    print("redrawing wordcloud...")
    n_float = float(n_selection / 100)
    print("got time window:", str(time_values))
    print("got n_selection:", str(n_selection), str(n_float))
    # sample the dataset according to the slider
    local_df = sample_data(GLOBAL_DF, n_float)
    if time_values is not None:
        time_values = time_slider_to_date(time_values)
        local_df = local_df[
            (local_df["Date"] >= time_values[0])
            & (local_df["Date"] <= time_values[1])
        ]
    if selected_themes:
        local_df = local_df[local_df["Theme"] == selected_themes]
        #add_stopwords(selected_themes)
    return local_df


def make_marks_time_slider(mini, maxi):
    """
    A helper function to generate a dictionary that should look something like:
    {1420066800: '2015', 1427839200: 'Q2', 1435701600: 'Q3', 1443650400: 'Q4',
    1451602800: '2016', 1459461600: 'Q2', 1467324000: 'Q3', 1475272800: 'Q4',
     1483225200: '2017', 1490997600: 'Q2', 1498860000: 'Q3', 1506808800: 'Q4'}
    """
    step = relativedelta.relativedelta(months=+12)
    start = datetime(year=2011, month=1, day=1)
    #thirty=[4,6,9,11]
    #if maxi.month==2:
    end = datetime(year=2020, month=2, day=28)
    #elif maxi.month in thirty:
    #	end = datetime(year=maxi.year, month=maxi.month, day=30)
    #else:
    #	end = datetime(year=maxi.year, month=maxi.month, day=31)

    ret = {}

    current = start
    while current <= end:
        current_str = int(current.timestamp())
#        if current.month == 1:
        ret[current_str] = {
            "label": str(current.year),
            "style": {"font-weight": "bold", "font-size": 7},
        }
#        elif current.month == 4:
#            ret[current_str] = {
#                "label": "Q2",
#                "style": {"font-weight": "lighter", "font-size": 5},
#            }
#        elif current.month == 7:
#            ret[current_str] = {
#                "label": "Q3",
#                "style": {"font-weight": "lighter", "font-size": 5},
#            }
#        elif current.month == 10:
#            ret[current_str] = {
#                "label": "Q4",
#                "style": {"font-weight": "lighter", "font-size": 5},
#            }
#        else:
#            pass
        current += step
    # print(ret)
    return ret


def time_slider_to_date(time_values):
    """ TODO """
    min_date = datetime.fromtimestamp(time_values[0]).strftime("%c")
    max_date = datetime.fromtimestamp(time_values[1]).strftime("%c")
    print("Converted time_values: ")
    print("\tmin_date:", time_values[0], "to: ", min_date)
    print("\tmax_date:", time_values[1], "to: ", max_date)
    return [min_date, max_date]


def make_options_themes_drop(values):
    """
    Helper function to generate the data format the dropdown dash component wants
    """
    ret = []
    for value in values:
        ret.append({"label": value, "value": value})
    return ret

def plotly_wordcloud(data_frame):
    """A wonderful function that returns figure data for three equally
    wonderful plots: wordcloud, frequency histogram and treemap"""
    complaints_text = list(data_frame["Article Text"].dropna().values)

    if len(complaints_text) < 1:
        return {}, {}, {}

    # join all documents in corpus
    text = " ".join(list(complaints_text))
    
    stopwords_wc = set(STOPWORDS)
    tokens = nltk.word_tokenize(text)
    text1 = nltk.Text(tokens)
    text_content = [word for word in text1 if word not in stopwords_wc]
    nltk_tokens = nltk.word_tokenize(text)  
    bigrams_list = list(nltk.bigrams(text_content))
    dictionary2 = [' '.join(tup) for tup in bigrams_list]
    vectorizer = CountVectorizer(ngram_range=(2, 2))
    bag_of_words = vectorizer.fit_transform(dictionary2)
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in vectorizer.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
    words_dict = dict(words_freq)
    WC_height = 1000
    WC_width = 1500
    WC_max_words = 200
    word_cloud = WordCloud(max_words=WC_max_words, height=WC_height, width=WC_width,stopwords=stopwords_wc)
    word_cloud.generate_from_frequencies(words_dict)

    #word_cloud = WordCloud(stopwords=set(STOPWORDS), max_words=100, max_font_size=90)
    #word_cloud.generate(text)

    word_list = []
    freq_list = []
    fontsize_list = []
    position_list = []
    orientation_list = []
    color_list = []

    for (word, freq), fontsize, position, orientation, color in word_cloud.layout_:
        word_list.append(word)
        freq_list.append(freq)
        fontsize_list.append(fontsize)
        position_list.append(position)
        orientation_list.append(orientation)
        color_list.append(color)

    # get the positions
    x_arr = []
    y_arr = []
    for i in position_list:
        x_arr.append(i[0])
        y_arr.append(i[1])

    # get the relative occurence frequencies
    new_freq_list = []
    for i in freq_list:
        new_freq_list.append(i * 80)

    trace = go.Scatter(
        x=x_arr,
        y=y_arr,
        textfont=dict(size=new_freq_list, color=color_list),
        hoverinfo="text",
        textposition="top center",
        hovertext=["{0} - {1}".format(w, f) for w, f in zip(word_list, freq_list)],
        mode="text",
        text=word_list,
    )

    layout = go.Layout(
        {
            "xaxis": {
                "showgrid": False,
                "showticklabels": False,
                "zeroline": False,
                "automargin": True,
                "range": [-100, 250],
            },
            "yaxis": {
                "showgrid": False,
                "showticklabels": False,
                "zeroline": False,
                "automargin": True,
                "range": [-100, 450],
            },
            "margin": dict(t=20, b=20, l=10, r=10, pad=4),
            "hovermode": "closest",
        }
    )

    wordcloud_figure_data = {"data": [trace], "layout": layout}
    word_list_top = word_list[:25]
    word_list_top.reverse()
    freq_list_top = freq_list[:25]
    freq_list_top.reverse()

    frequency_figure_data = {
        "data": [
            {
                "y": word_list_top,
                "x": freq_list_top,
                "type": "bar",
                "name": "",
                "orientation": "h",
            }
        ],
        "layout": {"height": "550", "margin": dict(t=20, b=20, l=100, r=20, pad=4)},
    }
    treemap_trace = go.Treemap(
        labels=word_list_top, parents=[""] * len(word_list_top), values=freq_list_top
    )
    treemap_layout = go.Layout({"margin": dict(t=10, b=10, l=5, r=5, pad=4)})
    treemap_figure = {"data": [treemap_trace], "layout": treemap_layout}
    return wordcloud_figure_data, frequency_figure_data, treemap_figure


"""
#  Page layout and contents

In an effort to clean up the code a bit, we decided to break it apart into
sections. For instance: LEFT_COLUMN is the input controls you see in that gray
box on the top left. The body variable is the overall structure which most other
sections go into. This just makes it ever so slightly easier to find the right
spot to add to or change without having to count too many brackets.
"""

NAVBAR = dbc.Navbar(
    children=[
        html.A(
            # Use row and col to control vertical alignment of logo / brand
            dbc.Row(
                [
                    dbc.Col(html.Img(src=PLOTLY_LOGO, height="30px")),
                    dbc.Col(
                        dbc.NavbarBrand("Articles around Aadhaar", className="ml-2")
                    ),
                ],
                align="center",
                no_gutters=True,
            ),
            href="https://plot.ly",
        )
    ],
    color="dark",
    dark=True,
    sticky="top",
)

LEFT_COLUMN = dbc.Jumbotron(
    [
        html.H4(children="Select theme & dataset size", className="display-5"),
        html.Hr(className="my-2"),
        html.Label("Select percentage of dataset", className="lead"),
        html.P(
            "(Lower is faster. Higher is more precise)",
            style={"fontSize": 10, "font-weight": "lighter"},
        ),
        dcc.Slider(
            id="n-selection-slider",
            min=1,
            max=100,
            step=1,
            marks={
                0: "0%",
                10: "",
                20: "20%",
                30: "",
                40: "40%",
                50: "",
                60: "60%",
                70: "",
                80: "80%",
                90: "",
                100: "100%",
            },
            value=20,
        ),
        html.Label("Select a theme", style={"marginTop": 50}, className="lead"),
        html.P(
            "(You can use the dropdown or click the barchart on the right)",
            style={"fontSize": 10, "font-weight": "lighter"},
        ),
        dcc.Dropdown(
            id="themes-drop", clearable=False, style={"marginBottom": 50, "font-size": 12}
        ),
        html.Label("Select time frame", className="lead"),
        html.Div(dcc.RangeSlider(id="time-window-slider"), style={"marginBottom": 50}),
    ]
)

LEFT_COLUMN_II = dbc.Jumbotron(
    [
        html.H4(children="Select theme & word", className="display-5"),
        html.Hr(className="my-2"),
        html.Label("Select a theme", style={"marginTop": 50}, className="lead"),
        dcc.Dropdown(
            id="theme-drop-ii", clearable=False, style={"marginBottom": 50, "font-size": 12}
        ),
        html.Label("Select a word", style={"marginTop": 50}, className="lead"),
        dcc.Dropdown(
            id="word-drop", clearable=False, style={"marginBottom": 50, "font-size": 12}
        ),
    ]
)

WORDCLOUD_PLOTS = [
    dbc.CardHeader(html.H5("Most frequently used two-word terms in themes")),
    dbc.Alert(
        "Not enough data to render these plots, please adjust the filters",
        id="no-data-alert",
        color="warning",
        style={"display": "none"},
    ),
    dbc.CardBody(
        [
            dbc.Row(
                [
                    dbc.Col(
                        dcc.Loading(
                            id="loading-frequencies",
                            children=[dcc.Graph(id="frequency_figure")],
                            type="default",
                        )
                    ),
                    dbc.Col(
                        [
                            dcc.Tabs(
                                id="tabs",
                                children=[
                                    dcc.Tab(
                                        label="Treemap",
                                        children=[
                                            dcc.Loading(
                                                id="loading-treemap",
                                                children=[dcc.Graph(id="themes-treemap")],
                                                type="default",
                                            )
                                        ],
                                    ),
                                    dcc.Tab(
                                        label="Wordcloud",
                                        children=[
                                            dcc.Loading(
                                                id="loading-wordcloud",
                                                children=[
                                                    dcc.Graph(id="themes-wordcloud")
                                                ],
                                                type="default",
                                            )
                                        ],
                                    ),
                                ],
                            )
                        ],
                        md=8,
                    ),
                ]
            )
        ]
    ),
]

TOP_THEMES_PLOT = [
    dbc.CardHeader(html.H5("Top 10 themes by number of articles")),
    dbc.CardBody(
        [
            dcc.Loading(
                id="loading-themes-hist",
                children=[
                    dbc.Alert(
                        "Not enough data to render this plot, please adjust the filters",
                        id="no-data-alert-themes",
                        color="warning",
                        style={"display": "none"},
                    ),
                    dcc.Graph(id="themes-sample"),
                ],
                type="default",
            )
        ],
        style={"marginTop": 0, "marginBottom": 0},
    ),
]

TOP_NGRAM_COMPS = [
    dbc.CardHeader(html.H5("Comparison of words for two themes")),
    dbc.CardBody(
        [
            dcc.Loading(
                id="loading-ngrams-comps",
                children=[
                    dbc.Alert(
                        "Something's gone wrong! Give us a moment, but try loading this page again if problem persists.",
                        id="no-data-alert-ngrams_comp",
                        color="warning",
                        style={"display": "none"},
                    ),
                    dbc.Row(
                        [
                            dbc.Col(html.P("Choose two themes to compare:"), md=12),
                            dbc.Col(
                                [
                                    dcc.Dropdown(
                                        id="ngrams-comp_1",
                                        options=[
                                            {"label": i, "value": i}
                                            for i in ngram_df.theme.unique()
                                        ],
                                        value="data_security",
                                    )
                                ],
                                md=6,
                            ),
                            dbc.Col(
                                [
                                    dcc.Dropdown(
                                        id="ngrams-comp_2",
                                        options=[
                                            {"label": i, "value": i}
                                            for i in ngram_df.theme.unique()
                                        ],
                                        value="aadhaar_based_schemes",
                                    )
                                ],
                                md=6,
                            ),
                        ]
                    ),
                    dcc.Graph(id="ngrams-comps"),
                ],
                type="default",
            )
        ],
        style={"marginTop": 0, "marginBottom": 0},
    ),
]

WORD_PLOT = [
    dbc.CardHeader(html.H5("Year-wise distrution of selected word in theme")),
    dbc.CardBody(
        [
            dcc.Loading(
                id="loading-word-hist",
                children=[
                    dbc.Alert(
                        "Not enough data to render this plot, please adjust the filters",
                        id="no-data-alert-word",
                        color="warning",
                        style={"display": "none"},
                    ),
                    dcc.Graph(id="word-sample"),
                ],
                type="default",
            )
        ],
        style={"marginTop": 0, "marginBottom": 0},
    ),
]

BODY = dbc.Container(
    [
#        dbc.Row([dbc.Col(dbc.Card(TOP_NGRAM_PLOT)),], style={"marginTop": 30}),
        dbc.Row(
            [
                dbc.Col(LEFT_COLUMN, md=4, align="center"),
                dbc.Col(dbc.Card(TOP_THEMES_PLOT), md=8),
            ],
            style={"marginTop": 30},
        ),
        dbc.Card(WORDCLOUD_PLOTS),
        dbc.Row([dbc.Col(dbc.Card(TOP_NGRAM_COMPS)),], style={"marginTop": 30}),
        dbc.Row(
            [
                dbc.Col(LEFT_COLUMN_II, md=4, align="center"),
                dbc.Col(dbc.Card(WORD_PLOT), md=8),
            ],
            style={"marginTop": 30},
        ),
#        dbc.Row([dbc.Col([dbc.Card(LDA_PLOTS)])], style={"marginTop": 50}),
    ],
    className="mt-12",
)

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server  # for Heroku deployment

app.layout = html.Div(children=[NAVBAR, BODY])

"""
#  Callbacks
"""

@app.callback(
    Output("ngrams-comps", "figure"),
    [Input("ngrams-comp_1", "value"), Input("ngrams-comp_2", "value")],
)
def comp_ngram_comparisons(comp_first, comp_second):
    comp_list = [comp_first, comp_second]
    temp_df = ngram_df[ngram_df.theme.isin(comp_list)]
    temp_df.loc[temp_df.theme == comp_list[-1], "value"] = -temp_df[
        temp_df.theme == comp_list[-1]
    ].value.values

    fig = px.bar(
        temp_df,
        title="Comparison: " + comp_first + " | " + comp_second,
        x="ngram",
        y="value",
        color="theme",
        template="plotly_white",
        color_discrete_sequence=px.colors.qualitative.Bold,
        labels={"theme": "Theme:", "ngram": "N-Gram"},
        hover_data="",
    )
    fig.update_layout(legend=dict(x=0.1, y=1.1), legend_orientation="h")
    fig.update_yaxes(title="", showticklabels=False)
    fig.data[0]["hovertemplate"] = fig.data[0]["hovertemplate"][:-14]
    return fig


@app.callback(
    [
        Output("time-window-slider", "marks"),
        Output("time-window-slider", "min"),
        Output("time-window-slider", "max"),
        Output("time-window-slider", "step"),
        Output("time-window-slider", "value"),
    ],
    [Input("n-selection-slider", "value")],
)
def populate_time_slider(value):
    """
    Depending on our dataset, we need to populate the time-slider
    with different ranges. This function does that and returns the
    needed data to the time-window-slider.
    """
    value += 0
    min_date = GLOBAL_DF["Date"].min()
    max_date = GLOBAL_DF["Date"].max()

    print(min_date)
    print(max_date)

    marks = make_marks_time_slider(min_date, max_date)
    min_epoch = list(marks.keys())[0]
    max_epoch = list(marks.keys())[-1]

    return (
        marks,
        min_epoch,
        max_epoch,
        (max_epoch - min_epoch) / (len(list(marks.keys())) * 3),
        [min_epoch, max_epoch],
    )


@app.callback(
    Output("themes-drop", "options"),
    [Input("time-window-slider", "value"), Input("n-selection-slider", "value")],
)
def populate_themes_dropdown(time_values, n_value):
    """ TODO """
    print("themes-drop: TODO USE THE TIME VALUES AND N-SLIDER TO LIMIT THE DATASET")
    if time_values is not None:
        pass
    n_value += 1
    themes_names, counts = get_complaint_count_by_company(GLOBAL_DF)
    counts.append(1)
    return make_options_themes_drop(themes_names)

@app.callback(
    Output("word-drop", "options"),
    [Input("time-window-slider", "value"), Input("n-selection-slider", "value")],
)
def populate_word_dropdown(time_values, n_value):
    """ TODO """
    print("word-drop: TODO USE THE TIME VALUES AND N-SLIDER TO LIMIT THE DATASET")
    if time_values is not None:
        pass
    n_value += 1
    words, counts = get_count_by_ngram(ngram_df)
    counts.append(1)
    return make_options_themes_drop(words)

@app.callback(
    Output("theme-drop-ii", "options"),
    [Input("time-window-slider", "value"), Input("n-selection-slider", "value")],
)
def populate_themes_dropdown(time_values, n_value):
    """ TODO """
    if time_values is not None:
        pass
    n_value += 1
    themes_names, counts = get_complaint_count_by_company(GLOBAL_DF)
    counts.append(1)
    return make_options_themes_drop(themes_names)

@app.callback(
    [Output("themes-sample", "figure"), Output("no-data-alert-themes", "style")],
    [Input("n-selection-slider", "value"), Input("time-window-slider", "value")],
)
def update_themes_sample_plot(n_value, time_values):
    """ TODO """
    print("redrawing sample...")
    print("\tn is:", n_value)
    print("\ttime_values is:", time_values)
    if time_values is None:
        return [{}, {"display": "block"}]
    n_float = float(n_value / 100)
    themes_sample_count = 10
    local_df = sample_data(GLOBAL_DF, n_float)
    min_date, max_date = time_slider_to_date(time_values)
    values_sample, counts_sample = calculate_themes_sample_data(
        local_df, themes_sample_count, [min_date, max_date]
    )
    data = [
        {
            "x": values_sample,
            "y": counts_sample,
            "text": values_sample,
            "textposition": "auto",
            "type": "bar",
            "name": "",
        }
    ]
    layout = {
        "autosize": False,
        "margin": dict(t=10, b=10, l=40, r=0, pad=4),
        "xaxis": {"showticklabels": False},
    }
    print("redrawing themes-sample...done")
    return [{"data": data, "layout": layout}, {"display": "none"}]

@app.callback(
    [
        Output("themes-wordcloud", "figure"),
        Output("frequency_figure", "figure"),
        Output("themes-treemap", "figure"),
        Output("no-data-alert", "style"),
    ],
    [
        Input("themes-drop", "value"),
        Input("time-window-slider", "value"),
        Input("n-selection-slider", "value"),
    ],
)
def update_wordcloud_plot(value_drop, time_values, n_selection):
    """ Callback to rerender wordcloud plot """
    local_df = make_local_df(value_drop, time_values, n_selection)
    wordcloud, frequency_figure, treemap = plotly_wordcloud(local_df)
    alert_style = {"display": "none"}
    if (wordcloud == {}) or (frequency_figure == {}) or (treemap == {}):
        alert_style = {"display": "block"}
    print("redrawing themes-wordcloud...done")
    return (wordcloud, frequency_figure, treemap, alert_style)

@app.callback(Output("themes-drop", "value"), [Input("themes-sample", "clickData")])
def update_themes_drop_on_click(value):
    """ TODO """
    if value is not None:
        selected_themes = value["points"][0]["x"]
        return selected_themes
    return "data_security"

@app.callback(
    [Output("word-sample", "figure"), Output("no-data-alert-word", "style")],
    [Input("theme-drop-ii", "value"), Input("word-drop", "value")],
)
def update_words_plot(theme, word):
    """ TODO """
    if word is None:
        return [{}, {"display": "block"}]
    themes_sample_count = 10
    #local_df = sample_data(GLOBAL_DF, n_float)
    values_sample, counts_sample = calculate_word_data(theme, word)
    data = [
        {
            "x": values_sample,
            "y": counts_sample,
            "text": values_sample,
            "textposition": "auto",
            "type": "bar",
            "name": "",
        }
    ]
    layout = {
        "autosize": False,
        "margin": dict(t=10, b=10, l=40, r=0, pad=4),
        "xaxis": {"showticklabels": False},
    }
    return [{"data": data, "layout": layout}, {"display": "none"}]

@app.callback(Output("ngrams-comp_1", "value"), [Input("themes-drop", "value")])
def update_themes_drop_on_click(value):
    """ TODO """
    if value is not None:
        selected_themes = value
        return selected_themes
    return "data_security"

@app.callback(Output("theme-drop-ii", "value"), [Input("ngrams-comp_1", "value")])
def update_themes_drop_on_click(value):
    """ TODO """
    if value is not None:
        selected_themes = value
        return selected_themes
    return "data_security"

@app.callback(Output("word-drop", "value"), [Input("ngrams-comps", "clickData")])
def update_themes_drop_on_click(value):
    """ TODO """
    if value is not None:
        selected_themes = value["points"][0]["x"]
        return selected_themes
    return "data"

if __name__ == "__main__":
    app.run_server(debug=True, threaded=True)

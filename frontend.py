from dash import Dash, dcc, html, Input, Output, callback, State
import dash_auth

# Define your username and password pairs
USERNAME_PASSWORD_PAIRS = {
    'admin': 'adminpassword'
}

app = Dash(__name__)
auth = dash_auth.BasicAuth(app, USERNAME_PASSWORD_PAIRS)

app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    html.Div(id='page-content')
])

index_page = html.Div([
    html.H1("Home Page"),
    html.Div([
        html.Img(src="http://localhost:5000/api/v1/video_feed"),
        html.Br(),
        html.H2('Person : Unkown'),
        html.H2('Confidence: 0%'),
        dcc.Link('Go to Admin Page', href='/admin')
    ])
])

admin_page = html.Div([
    html.H1('Admin Dashboard'),
    html.Div([
        html.H2('Add a Person'),
        html.Img(src="http://localhost:5000/api/v1/video_feed"),
        html.Div(id='add-container'),
        dcc.Input(id='person-name', type='text', placeholder='Enter person name'),
        html.Button('Submit', id='submit-btn', n_clicks=0),
        html.Div(id='output-container'),
        html.Br(),
        dcc.Link('Go back to Home', href='/')
    ])
])

# Update the index
@app.callback(Output('page-content', 'children'),
              [Input('url', 'pathname')])
def display_page(pathname):
    if pathname == '/admin':
        return admin_page
    else:
        return index_page

# Add person callback
@app.callback(
    Output('output-container', 'children'),
    [Input('submit-btn', 'n_clicks')],
    [State('person-name', 'value')])
def update_output(n_clicks, value):
    if n_clicks > 0:
        if value is None:
            return 'Please enter a name'
        import requests
        # This is a placeholder for how you might send data to your Flask API
        response = requests.post('http://localhost:5000/api/v1/add_person', json={'name': value, 'image': 'image_data_here'})
        if response.status_code == 201:
            return 'Person added successfully'
        else:
            return 'Failed to add person'
    return ''  # PreventUpdate can also be used here if no updates should be made

if __name__ == '__main__':
    app.run_server(debug=True)

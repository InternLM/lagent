import json
import os
from typing import Optional, Type

import requests

from lagent.actions.base_action import BaseAction
from lagent.actions.parser import BaseParser, JsonParser

DEFAULT_DESCRIPTION = dict(
    name='BINGMap',
    description='Plugin for looking up map information',
    api_list=[
        dict(
            name='get_distance',
            description='Get the distance between two locations in km.',
            parameters=[
                dict(
                    name='start',
                    type='STRING',
                    description='The start location.'),
                dict(
                    name='end', type='STRING', description='The end location.')
            ],
            required=['start', 'end'],
            return_data=[
                dict(name='distance', description='the distance in km.')
            ]),
        dict(
            name='get_route',
            description='Get the route between two locations in km.',
            parameters=[
                dict(
                    name='start',
                    type='STRING',
                    description='The start location.'),
                dict(
                    name='end', type='STRING', description='The end location.')
            ],
            required=['start', 'end'],
            return_data=[
                dict(
                    name='route', description='the route, a list of actions.')
            ]),
        dict(
            name='get_coordinates',
            description='Get the coordinates of a location.',
            parameters=[
                dict(
                    name='location',
                    type='STRING',
                    description='the location need to get coordinates.')
            ],
            required=['location'],
            return_data=[
                dict(
                    name='latitude',
                    description='the latitude of the location.'),
                dict(
                    name='longitude',
                    description='the longitude of the location.')
            ]),
        dict(
            name='search_nearby',
            description=
            'Search for places nearby a location, within a given radius, and return the results into a list. You can use either the places name or the latitude and longitude.',
            parameters=[
                dict(
                    name='search_term',
                    type='STRING',
                    description='the place name'),
                dict(
                    name='places',
                    type='STRING',
                    description='the name of the location.'),
                dict(
                    name='latitude',
                    type='FLOAT',
                    description='the latitude of the location.'),
                dict(
                    name='longitude',
                    type='FLOAT',
                    description='the longitude of the location.'),
                dict(
                    name='radius',
                    type='NUMBER',
                    description='radius in meters.')
            ],
            required=['search_term'],
            return_data=[
                dict(
                    name='places',
                    description=
                    'the list of places, each place is a dict with name and address, at most 5 places.'
                )
            ]),
    ])


class BINGMap(BaseAction):
    """BING Map plugin for looking up map information"""

    def __init__(self,
                 key: Optional[str] = None,
                 description: Optional[dict] = None,
                 parser: Type[BaseParser] = JsonParser,
                 enable: bool = True) -> None:
        super().__init__(description or DEFAULT_DESCRIPTION, parser, enable)
        key = os.environ.get('BING_MAP_KEY')
        if key is None:
            raise ValueError(
                'Please set BING Map API key either in the environment '
                'as BING_MAP_KEY or pass it as `key` parameter.')
        self.key = key
        self.base_url = 'http://dev.virtualearth.net/REST/V1/'

    def get_distance(self, **args):
        start, end = args['start'], args['end']
        # Request URL
        url = self.base_url + 'Routes/Driving?o=json&wp.0=' + start + '&wp.1=' + end + '&key=' + self.key
        # GET request
        r = requests.get(url)
        # TODO check request status?
        data = json.loads(r.text)
        # Extract route information
        route = data['resourceSets'][0]['resources'][0]
        # Extract distance in miles
        distance = route['travelDistance']
        return dict(distance=distance)

    def get_route(self, **args):
        start, end = args['start'], args['end']
        # Request URL
        url = self.base_url + 'Routes/Driving?o=json&wp.0=' + start + '&wp.1=' + end + '&key=' + self.key
        # GET request
        r = requests.get(url)
        data = json.loads(r.text)
        # Extract route information
        route = data['resourceSets'][0]['resources'][0]
        itinerary = route['routeLegs'][0]['itineraryItems']
        # Extract route text information
        route_text = []
        for item in itinerary:
            if 'instruction' in item:
                route_text.append(item['instruction']['text'])
        return dict(route=route_text)

    def get_coordinates(self, **args):
        location = args['location']
        url = self.base_url + 'Locations'
        params = {'query': location, 'key': self.key}
        response = requests.get(url, params=params)
        json_data = response.json()
        coordinates = json_data['resourceSets'][0]['resources'][0]['point'][
            'coordinates']
        return dict(latitude=coordinates[0], longitude=coordinates[1])

    def search_nearby(self, **args):  #  radius in meters
        if 'latitude' not in args: args['latitude'] = 0.0
        if 'longitude' not in args: args['longitude'] = 0.0
        if 'places' not in args: args['places'] = 'unknown'
        if 'radius' not in args: args['radius'] = 5000
        search_term = args['search_term']
        latitude = args['latitude']
        longitude = args['longitude']
        places = args['places']
        radius = args['radius']
        url = self.base_url + 'LocalSearch'
        if places != 'unknown':
            pos = self.get_coordinates(**{'location': places})
            latitude, longitude = pos[1]['latitude'], pos[1]['longitude']
        # Build the request query string
        params = {
            'query': search_term,
            'userLocation': f'{latitude},{longitude}',
            'radius': radius,
            'key': self.key
        }
        # Make the request
        response = requests.get(url, params=params)
        # Parse the response
        response_data = json.loads(response.content)
        # Get the results
        results = response_data['resourceSets'][0]['resources']
        addresses = []
        for result in results:
            name = result['name']
            address = result['Address']['formattedAddress']
            addresses.append(dict(name=name, address=address))
            if len(addresses) == 5:
                break
        return dict(place=addresses)

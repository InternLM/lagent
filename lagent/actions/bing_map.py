# flake8: noqa: E501
import json
import os
from typing import Optional, Type

import requests

from lagent.actions.base_action import BaseAction, tool_api
from lagent.actions.parser import BaseParser, JsonParser


class BINGMap(BaseAction):
    """BING Map plugin for looking up map information."""

    def __init__(self,
                 key: Optional[str] = None,
                 description: Optional[dict] = None,
                 parser: Type[BaseParser] = JsonParser,
                 enable: bool = True) -> None:
        super().__init__(description, parser, enable)
        key = os.environ.get('BING_MAP_KEY', key)
        if key is None:
            raise ValueError(
                'Please set BING Map API key either in the environment '
                'as BING_MAP_KEY or pass it as `key` parameter.')
        self.key = key
        self.base_url = 'http://dev.virtualearth.net/REST/V1/'

    @tool_api(explode_return=True)
    def get_distance(self, start: str, end: str) -> dict:
        """Get the distance between two locations in km.

        Args:
            start (:class:`str`): The start location
            end (:class:`str`): The end location

        Returns:
            :class:`dict`: distance information
                * distance (str): the distance in km.
        """
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

    @tool_api(explode_return=True)
    def get_route(self, start: str, end: str) -> dict:
        """Get the route between two locations in km.

        Args:
            start (:class:`str`): The start location
            end (:class:`str`): The end location

        Returns:
            :class:`dict`: route information
                * route (list): the route, a list of actions.
        """
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

    @tool_api(explode_return=True)
    def get_coordinates(self, location: str) -> dict:
        """Get the coordinates of a location.

        Args:
            location (:class:`str`): the location need to get coordinates.

        Returns:
            :class:`dict`: coordinates information
                * latitude (float): the latitude of the location.
                * longitude (float): the longitude of the location.
        """
        url = self.base_url + 'Locations'
        params = {'query': location, 'key': self.key}
        response = requests.get(url, params=params)
        json_data = response.json()
        coordinates = json_data['resourceSets'][0]['resources'][0]['point'][
            'coordinates']
        return dict(latitude=coordinates[0], longitude=coordinates[1])

    @tool_api(explode_return=True)
    def search_nearby(self,
                      search_term: str,
                      places: str = 'unknown',
                      latitude: float = 0.0,
                      longitude: float = 0.0,
                      radius: int = 5000) -> dict:
        """Search for places nearby a location, within a given radius, and return the results into a list. You can use either the places name or the latitude and longitude.

        Args:
            search_term (:class:`str`): the place name.
            places (:class:`str`): the name of the location. Defaults to ``'unknown'``.
            latitude (:class:`float`): the latitude of the location. Defaults to ``0.0``.
            longitude (:class:`float`): the longitude of the location. Defaults to ``0.0``.
            radius (:class:`int`): radius in meters. Defaults to ``5000``.

        Returns:
            :class:`dict`: places information
                * places (list): the list of places, each place is a dict with name and address, at most 5 places.
        """
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

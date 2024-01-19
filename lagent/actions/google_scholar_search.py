import json
import os
from typing import Optional, Type

from serpapi import GoogleSearch

from lagent.actions.base_action import BaseAction
from lagent.schema import ActionReturn, ActionStatusCode
from .parser import BaseParser, JsonParser

DEFAULT_DESCRIPTION = dict(
    name='GoogleScholar',
    description='Plugin for google scholar search',
    api_list=[{
        'name':
        'search_google_scholar',
        'description':
        'Search for scholarly articles based on a query according to the google scholar',
        'parameters': [
            {
                'name': 'query',
                'description': 'The query to search for.',
                'type': 'STRING'
            },
            {
                'name': 'cites',
                'description':
                'The unique ID of an article for triggering "Cited By" searches',
                'type': 'STRING'
            },
            {
                'name': 'as_ylo',
                'description':
                'The starting year for results (e.g., if as_ylo=2018, results before this year will be omitted)',
                'type': 'NUMBER'
            },
            {
                'name': 'as_yhi',
                'description':
                'The ending year for results (e.g., if as_yhi=2018, results after this year will be omitted)',
                'type': 'NUMBER'
            },
            {
                'name': 'scisbd',
                'description':
                'Defines articles added in the last year, sorted by date. It can be set to 1 to include only abstracts, or 2 to include everything',
                'type': 'NUMBER'
            },
            {
                'name': 'cluster',
                'description':
                'The unique ID of an article for triggering "All Versions" searches',
                'type': 'STRING'
            },
            {
                'name': 'hl',
                'description':
                'The language to use for the Google Scholar search',
                'type': 'STRING'
            },
            {
                'name': 'lr',
                'description':
                'One or multiple languages to limit the search to',
                'type': 'STRING'
            },
            {
                'name': 'start',
                'description':
                'The result offset for pagination (0 is the first page of results, 10 is the 2nd page, etc.)',
                'type': 'NUMBER'
            },
            {
                'name': 'num',
                'description':
                'The maximum number of results to return, limited to 20',
                'type': 'NUMBER'
            },
            {
                'name': 'as_sdt',
                'description':
                'Can be used either as a search type or a filter',
                'type': 'STRING'
            },
            {
                'name': 'safe',
                'description': 'The level of filtering for adult content',
                'type': 'STRING'
            },
            {
                'name': 'filter',
                'description':
                "Defines if the filters for 'Similar Results' and 'Omitted Results' are on or off",
                'type': 'STRING'
            },
            {
                'name': 'as_vis',
                'description': 'Defines whether to include citations or not',
                'type': 'STRING'
            },
        ],
        'required': ['query'],
        'return_data': [{
            'name':
            'title',
            'description':
            'a list of the titles of the three selected papers'
        }, {
            'name':
            'cited_by',
            'description':
            'a list of the citation numbers of the three selected papers'
        }, {
            'name':
            'organic_id',
            'description':
            'a list of the organic results\' ids of the three selected papers'
        }, {
            'name': 'snippets',
            'description': 'snippets of the papers'
        }, {
            'name':
            'pub_info',
            'description':
            'publication information of selected papers'
        }]
    }, {
        'name':
        'get_author_information',
        'description':
        'Search for an author\'s information by author\'s id provided by get_author_id.',
        'parameters': [{
            'name': 'author_id',
            'description': 'Required. The ID of an author.',
            'type': 'STRING'
        }, {
            'name': 'hl',
            'description':
            "The language to use for the Google Scholar Author search. Default is 'en'.",
            'type': 'STRING'
        }, {
            'name': 'view_op',
            'description': 'Used for viewing specific parts of a page.',
            'type': 'STRING'
        }, {
            'name': 'sort',
            'description': 'Used for sorting and refining articles.',
            'type': 'STRING'
        }, {
            'name': 'citation_id',
            'description': 'Used for retrieving individual article citation.',
            'type': 'STRING'
        }, {
            'name': 'start',
            'description': 'Defines the result offset. Default is 0.',
            'type': 'NUMBER'
        }, {
            'name': 'num',
            'description':
            'Defines the number of results to return. Default is 20.',
            'type': 'NUMBER'
        }, {
            'name': 'no_cache',
            'description':
            'Forces SerpApi to fetch the results even if a cached version is already present. Default is False.',
            'type': 'BOOLEAN'
        }, {
            'name': 'async_req',
            'description':
            'Defines the way you want to submit your search to SerpApi. Default is False.',
            'type': 'BOOLEAN'
        }, {
            'name': 'output',
            'description':
            "Defines the final output you want. Default is 'json'.",
            'type': 'STRING'
        }],
        'required': ['author_id'],
        'return_data': [{
            'name': 'name',
            'description': "author's name"
        }, {
            'name': 'affliation',
            'description': 'the affliation of the author'
        }, {
            'name': 'articles',
            'description': 'at most 3 articles by the author'
        }, {
            'name': 'website',
            'description': "the author's homepage url"
        }]
    }, {
        'name':
        'get_citation_format',
        'description':
        'Function to get MLA citation format by an identification of organic_result\'s id provided by search_google_scholar.',
        'parameters': [{
            'name': 'q',
            'description':
            'ID of an individual Google Scholar organic search result.',
            'type': 'STRING'
        }, {
            'name': 'engine',
            'description':
            "Set to 'google_scholar_cite' to use the Google Scholar API engine. Defaults to 'google_scholar_cite'.",
            'type': 'STRING'
        }, {
            'name': 'no_cache',
            'description':
            'If set to True, will force SerpApi to fetch the Google Scholar Cite results even if a cached version is already present. Defaults to None.',
            'type': 'BOOLEAN'
        }, {
            'name': 'async_',
            'description':
            'If set to True, will submit search to SerpApi and retrieve results later. Defaults to None.',
            'type': 'BOOLEAN'
        }, {
            'name': 'output',
            'description':
            "Final output format. Set to 'json' to get a structured JSON of the results, or 'html' to get the raw html retrieved. Defaults to 'json'.",
            'type': 'STRING'
        }],
        'required': ['q'],
        'return_data': [{
            'name': 'authors',
            'description': 'the authors of the article'
        }, {
            'name': 'citation',
            'description': 'the citation format of the article'
        }]
    }, {
        'name':
        'get_author_id',
        'description':
        'The getAuthorId function is used to get the author\'s id by his or her name.',
        'parameters': [{
            'name': 'mauthors',
            'description': 'Defines the author you want to search for.',
            'type': 'STRING'
        }, {
            'name': 'hl',
            'description':
            "Defines the language to use for the Google Scholar Profiles search.It's a two-letter language code. (e.g., 'en' for English, 'es' for Spanish, or 'fr' for French). Defaults to 'en'.",
            'type': 'STRING'
        }, {
            'name': 'after_author',
            'description':
            'Defines the next page token. It is used for retrieving the next page results. The parameter has the precedence over before_author parameter. Defaults to None.',
            'type': 'STRING'
        }, {
            'name': 'before_author',
            'description':
            'Defines the previous page token. It is used for retrieving the previous page results. Defaults to None.',
            'type': 'STRING'
        }, {
            'name': 'no_cache',
            'description':
            'Will force SerpApi to fetch the Google Scholar Profiles results even if a cached version is already present. Defaults to False.',
            'type': 'BOOLEAN'
        }, {
            'name': '_async',
            'description':
            'Defines the way you want to submit your search to SerpApi. Defaults to False.',
            'type': 'BOOLEAN'
        }, {
            'name': 'output',
            'description':
            "Defines the final output you want. It can be set to 'json' (default) to get a structured JSON of the results, or 'html' to get the raw html retrieved. Defaults to 'json'.",
            'type': 'STRING'
        }],
        'required': ['mauthors'],
        'return_data': [{
            'name': 'author_id',
            'description': 'the author_id of the author'
        }],
    }])


class GoogleScholar(BaseAction):
    """Wrapper around the Serper.dev Google Search API.

    To use, you should pass your serper API key to the constructor.

    Code is modified from lang-chain GoogleSerperAPIWrapper
    (https://github.com/langchain-ai/langchain/blob/ba5f
    baba704a2d729a4b8f568ed70d7c53e799bb/libs/langchain/
    langchain/utilities/google_serper.py)

    Args:
        api_key (str): API KEY to use serper google search API,
            You can create a free API key at https://serper.dev.
        description (dict): The description of the action. Defaults to 
            :py:data:`~DEFAULT_DESCRIPTION`.
        parser (Type[BaseParser]): The parser class to process the
            action's inputs and outputs. Defaults to :class:`JsonParser`.
        enable (bool, optional): Whether the action is enabled. Defaults to
            True.
    """

    def __init__(self,
                 api_key: Optional[str] = None,
                 description: Optional[dict] = None,
                 parser: Type[BaseParser] = JsonParser,
                 enable: bool = True):
        super().__init__(description or DEFAULT_DESCRIPTION, parser, enable)
        api_key = os.environ.get('SERPER_API_KEY', api_key)
        if api_key is None:
            raise ValueError(
                'Please set Serper API key either in the environment '
                'as SERPER_API_KEY or pass it as `api_key` parameter.')
        self.api_key = api_key

    def search_google_scholar(self, **param) -> ActionReturn:
        params = {
            'q': param['query'],
            'engine': 'google_scholar',
            'api_key': self.api_key
        }
        search = GoogleSearch(params)
        try:
            r = search.get_dict()
            results = r['organic_results']
            title = []
            snippets = []
            cited_by = []
            organic_id = []
            pub_info = []
            for item in results[:3]:
                title.append(item['title'])
                pub_info.append(item['publication_info']['summary'])
                citation = item['inline_links'].get('cited_by', {'total': ''})
                cited_by.append(citation['total'])
                snippets.append(item['snippet'])
                organic_id.append(item['result_id'])
            res = dict(
                title=title,
                cited_by=cited_by,
                organic_id=organic_id,
                snippets=snippets)
            return ActionReturn(
                param, result={'text': json.dumps(res, ensure_ascii=False)})
        except Exception as e:
            return ActionReturn(
                param, errmsg=str(e), state=ActionStatusCode.HTTP_ERROR)

    def get_author_information(self, **param) -> ActionReturn:
        params = {
            'engine': 'google_scholar_author',
            'author_id': param['author_id'],
            'api_key': self.api_key
        }
        try:
            search = GoogleSearch(params)
            results = search.get_dict()
            author = results['author']
            articles = results.get('articles', [])
            author_info = dict(
                name=author['name'],
                affiliations=author.get('affiliations', ''),
                website=author.get('website', ''),
                articles=[
                    dict(title=article['title'], authors=article['authors'])
                    for article in articles[:3]
                ])
            return ActionReturn(
                param,
                result={'text': json.dumps(author_info, ensure_ascii=False)})
        except Exception as e:
            return ActionReturn(
                param, errmsg=str(e), state=ActionStatusCode.HTTP_ERROR)

    def get_citation_format(self, **param) -> ActionReturn:
        params = {
            'q': param['q'],
            'engine': 'google_scholar_cite',
            'api_key': self.api_key
        }
        try:
            search = GoogleSearch(params)
            results = search.get_dict()
            citation = results['citations']
            citation_info = citation[0]['snippet']
            return ActionReturn(
                param,
                result={'text': json.dumps(citation_info, ensure_ascii=False)})
        except Exception as e:
            return ActionReturn(
                param, errmsg=str(e), state=ActionStatusCode.HTTP_ERROR)

    def get_author_id(self, **param) -> ActionReturn:
        params = {
            'mauthors': param['mauthors'],
            'engine': 'google_scholar_profiles',
            'api_key': self.api_key
        }
        try:
            search = GoogleSearch(params)
            results = search.get_dict()
            profile = results['profiles']
            author_info = dict(author_id=profile[0]['author_id'])
            return ActionReturn(
                param,
                result={'text': json.dumps(author_info, ensure_ascii=False)})
        except Exception as e:
            return ActionReturn(
                param, errmsg=str(e), state=ActionStatusCode.HTTP_ERROR)

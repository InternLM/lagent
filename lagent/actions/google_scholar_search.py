# flake8: noqa: E501
import os
from typing import Optional, Type

from lagent.actions.base_action import BaseAction, tool_api
from lagent.schema import ActionReturn, ActionStatusCode
from .parser import BaseParser, JsonParser


class GoogleScholar(BaseAction):
    """Plugin for google scholar search.

    Args:
        api_key (str): API KEY to use serper google search API,
            You can create a free API key at https://serper.dev.
        description (dict): The description of the action. Defaults to ``None``.
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
        super().__init__(description, parser, enable)
        api_key = os.environ.get('SERPER_API_KEY', api_key)
        if api_key is None:
            raise ValueError(
                'Please set Serper API key either in the environment '
                'as SERPER_API_KEY or pass it as `api_key` parameter.')
        self.api_key = api_key

    @tool_api(explode_return=True)
    def search_google_scholar(
        self,
        query: str,
    ) -> dict:
        """Search for scholarly articles based on a query according to the google scholar.

        Args:
            query (str): The query to search for.

        Returns:
            :class:`dict`: article information
                - title: a list of the titles of the three selected papers
                - cited_by: a list of the citation numbers of the three selected papers
                - organic_id: a list of the organic results' ids of the three selected papers
                - pub_info: publication information of selected papers
        """
        from serpapi import GoogleSearch
        params = {
            'q': query,
            'engine': 'google_scholar',
            'api_key': self.api_key,
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
            return dict(
                title=title,
                cited_by=cited_by,
                organic_id=organic_id,
                snippets=snippets)
        except Exception as e:
            return ActionReturn(
                errmsg=str(e), state=ActionStatusCode.HTTP_ERROR)

    @tool_api(explode_return=True)
    def get_author_information(
        self,
        author_id: str,
    ) -> dict:
        """Search for an author's information by author's id provided by get_author_id.

        Args:
            author_id (str): Required. The ID of an author.

        Returns:
            :class:`dict`: author information
                * name: author's name
                * affliation: the affliation of the author
                * articles: at most 3 articles by the author
                * website: the author's homepage url
        """
        from serpapi import GoogleSearch
        params = {
            'engine': 'google_scholar_author',
            'author_id': author_id,
            'api_key': self.api_key,
        }
        try:
            search = GoogleSearch(params)
            results = search.get_dict()
            author = results['author']
            articles = results.get('articles', [])
            return dict(
                name=author['name'],
                affiliations=author.get('affiliations', ''),
                website=author.get('website', ''),
                articles=[
                    dict(title=article['title'], authors=article['authors'])
                    for article in articles[:3]
                ])
        except Exception as e:
            return ActionReturn(
                errmsg=str(e), state=ActionStatusCode.HTTP_ERROR)

    @tool_api(explode_return=True)
    def get_citation_format(
        self,
        q: str,
    ) -> dict:
        """Function to get MLA citation format by an identification of organic_result's id provided by search_google_scholar.

        Args:
            q (str): ID of an individual Google Scholar organic search result.

        Returns:
            :class:`dict`: citation format
                * authors: the authors of the article
                * citation: the citation format of the article
        """
        from serpapi import GoogleSearch
        params = {
            'q': q,
            'engine': 'google_scholar_cite',
            'api_key': self.api_key,
        }
        try:
            search = GoogleSearch(params)
            results = search.get_dict()
            citation = results['citations']
            citation_info = citation[0]['snippet']
            return citation_info
        except Exception as e:
            return ActionReturn(
                errmsg=str(e), state=ActionStatusCode.HTTP_ERROR)

    @tool_api(explode_return=True)
    def get_author_id(self, mauthors: str) -> dict:
        """The getAuthorId function is used to get the author's id by his or her name.

        Args:
            mauthors (str): Defines the author you want to search for.

        Returns:
            :class:`dict`: author id
                * author_id: the author_id of the author
        """
        from serpapi import GoogleSearch
        params = {'mauthors': mauthors}
        try:
            search = GoogleSearch(params)
            results = search.get_dict()
            profile = results['profiles']
            author_info = dict(author_id=profile[0]['author_id'])
            return author_info
        except Exception as e:
            return ActionReturn(
                errmsg=str(e), state=ActionStatusCode.HTTP_ERROR)

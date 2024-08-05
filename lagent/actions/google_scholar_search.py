# flake8: noqa: E501
import os
from typing import Optional, Type

from aioify import aioify

from lagent.actions.base_action import AsyncActionMixin, BaseAction, tool_api
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
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        description: Optional[dict] = None,
        parser: Type[BaseParser] = JsonParser,
    ):
        super().__init__(description, parser)
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
        cites: Optional[str] = None,
        as_ylo: Optional[int] = None,
        as_yhi: Optional[int] = None,
        scisbd: Optional[int] = None,
        cluster: Optional[str] = None,
        hl: Optional[str] = None,
        lr: Optional[str] = None,
        start: Optional[int] = None,
        num: Optional[int] = None,
        as_sdt: Optional[str] = None,
        safe: Optional[str] = None,
        filter: Optional[str] = None,
        as_vis: Optional[str] = None,
    ) -> dict:
        """Search for scholarly articles based on a query according to the google scholar.

        Args:
            query (str): The query to search for.
            cites (Optional[str]): The unique ID of an article for triggering "Cited By" searches.
            as_ylo (Optional[int]): The starting year for results (e.g., if as_ylo=2018, results before this year will be omitted).
            as_yhi (Optional[int]): The ending year for results (e.g., if as_yhi=2018, results after this year will be omitted).
            scisbd (Optional[int]): Defines articles added in the last year, sorted by date. It can be set to 1 to include only abstracts, or 2 to include everything.
            cluster (Optional[str]): The unique ID of an article for triggering "All Versions" searches.
            hl (Optional[str]): The language to use for the Google Scholar search.
            lr (Optional[str]): One or multiple languages to limit the search to.
            start (Optional[int]): The result offset for pagination (0 is the first page of results, 10 is the 2nd page, etc.)
            num (Optional[int]): The maximum number of results to return, limited to 20.
            as_sdt (Optional[str]): Can be used either as a search type or a filter.
            safe (Optional[str]): The level of filtering for adult content.
            filter (Optional[str]): Defines if the filters for 'Similar Results' and 'Omitted Results' are on or off.
            as_vis (Optional[str]): Defines whether to include citations or not.

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
            'cites': cites,
            'as_ylo': as_ylo,
            'as_yhi': as_yhi,
            'scisbd': scisbd,
            'cluster': cluster,
            'hl': hl,
            'lr': lr,
            'start': start,
            'num': num,
            'as_sdt': as_sdt,
            'safe': safe,
            'filter': filter,
            'as_vis': as_vis
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
    def get_author_information(self,
                               author_id: str,
                               hl: Optional[str] = None,
                               view_op: Optional[str] = None,
                               sort: Optional[str] = None,
                               citation_id: Optional[str] = None,
                               start: Optional[int] = None,
                               num: Optional[int] = None,
                               no_cache: Optional[bool] = None,
                               async_req: Optional[bool] = None,
                               output: Optional[str] = None) -> dict:
        """Search for an author's information by author's id provided by get_author_id.

        Args:
            author_id (str): Required. The ID of an author.
            hl (Optional[str]): The language to use for the Google Scholar Author search. Default is 'en'.
            view_op (Optional[str]): Used for viewing specific parts of a page.
            sort (Optional[str]): Used for sorting and refining articles.
            citation_id (Optional[str]): Used for retrieving individual article citation.
            start (Optional[int]): Defines the result offset. Default is 0.
            num (Optional[int]): Defines the number of results to return. Default is 20.
            no_cache (Optional[bool]): Forces SerpApi to fetch the results even if a cached version is already present. Default is False.
            async_req (Optional[bool]): Defines the way you want to submit your search to SerpApi. Default is False.
            output (Optional[str]): Defines the final output you want. Default is 'json'.

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
            'hl': hl,
            'view_op': view_op,
            'sort': sort,
            'citation_id': citation_id,
            'start': start,
            'num': num,
            'no_cache': no_cache,
            'async': async_req,
            'output': output
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
    def get_citation_format(self,
                            q: str,
                            no_cache: Optional[bool] = None,
                            async_: Optional[bool] = None,
                            output: Optional[str] = 'json') -> dict:
        """Function to get MLA citation format by an identification of organic_result's id provided by search_google_scholar.

        Args:
            q (str): ID of an individual Google Scholar organic search result.
            no_cache (Optional[bool]): If set to True, will force SerpApi to fetch the Google Scholar Cite results even if a cached version is already present. Defaults to None.
            async_ (Optional[bool]): If set to True, will submit search to SerpApi and retrieve results later. Defaults to None.
            output (Optional[str]): Final output format. Set to 'json' to get a structured JSON of the results, or 'html' to get the raw html retrieved. Defaults to 'json'.

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
            'no_cache': no_cache,
            'async': async_,
            'output': output
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
    def get_author_id(self,
                      mauthors: str,
                      hl: Optional[str] = 'en',
                      after_author: Optional[str] = None,
                      before_author: Optional[str] = None,
                      no_cache: Optional[bool] = False,
                      _async: Optional[bool] = False,
                      output: Optional[str] = 'json') -> dict:
        """The getAuthorId function is used to get the author's id by his or her name.

        Args:
            mauthors (str): Defines the author you want to search for.
            hl (Optional[str]): Defines the language to use for the Google Scholar Profiles search. It's a two-letter language code. (e.g., 'en' for English, 'es' for Spanish, or 'fr' for French). Defaults to 'en'.
            after_author (Optional[str]): Defines the next page token. It is used for retrieving the next page results. The parameter has the precedence over before_author parameter. Defaults to None.
            before_author (Optional[str]): Defines the previous page token. It is used for retrieving the previous page results. Defaults to None.
            no_cache (Optional[bool]): Will force SerpApi to fetch the Google Scholar Profiles results even if a cached version is already present. Defaults to False.
            _async (Optional[bool]): Defines the way you want to submit your search to SerpApi. Defaults to False.
            output (Optional[str]): Defines the final output you want. It can be set to 'json' (default) to get a structured JSON of the results, or 'html' to get the raw html retrieved. Defaults to 'json'.

        Returns:
            :class:`dict`: author id
                * author_id: the author_id of the author
        """
        from serpapi import GoogleSearch
        params = {
            'mauthors': mauthors,
            'engine': 'google_scholar_profiles',
            'api_key': self.api_key,
            'hl': hl,
            'after_author': after_author,
            'before_author': before_author,
            'no_cache': no_cache,
            'async': _async,
            'output': output
        }
        try:
            search = GoogleSearch(params)
            results = search.get_dict()
            profile = results['profiles']
            author_info = dict(author_id=profile[0]['author_id'])
            return author_info
        except Exception as e:
            return ActionReturn(
                errmsg=str(e), state=ActionStatusCode.HTTP_ERROR)


class AsyncGoogleScholar(AsyncActionMixin, GoogleScholar):
    """Plugin for google scholar search.

    Args:
        api_key (str): API KEY to use serper google search API,
            You can create a free API key at https://serper.dev.
        description (dict): The description of the action. Defaults to ``None``.
        parser (Type[BaseParser]): The parser class to process the
            action's inputs and outputs. Defaults to :class:`JsonParser`.
    """

    @tool_api(explode_return=True)
    @aioify
    def search_google_scholar(
        self,
        query: str,
        cites: Optional[str] = None,
        as_ylo: Optional[int] = None,
        as_yhi: Optional[int] = None,
        scisbd: Optional[int] = None,
        cluster: Optional[str] = None,
        hl: Optional[str] = None,
        lr: Optional[str] = None,
        start: Optional[int] = None,
        num: Optional[int] = None,
        as_sdt: Optional[str] = None,
        safe: Optional[str] = None,
        filter: Optional[str] = None,
        as_vis: Optional[str] = None,
    ) -> dict:
        """Search for scholarly articles based on a query according to the google scholar.

        Args:
            query (str): The query to search for.
            cites (Optional[str]): The unique ID of an article for triggering "Cited By" searches.
            as_ylo (Optional[int]): The starting year for results (e.g., if as_ylo=2018, results before this year will be omitted).
            as_yhi (Optional[int]): The ending year for results (e.g., if as_yhi=2018, results after this year will be omitted).
            scisbd (Optional[int]): Defines articles added in the last year, sorted by date. It can be set to 1 to include only abstracts, or 2 to include everything.
            cluster (Optional[str]): The unique ID of an article for triggering "All Versions" searches.
            hl (Optional[str]): The language to use for the Google Scholar search.
            lr (Optional[str]): One or multiple languages to limit the search to.
            start (Optional[int]): The result offset for pagination (0 is the first page of results, 10 is the 2nd page, etc.)
            num (Optional[int]): The maximum number of results to return, limited to 20.
            as_sdt (Optional[str]): Can be used either as a search type or a filter.
            safe (Optional[str]): The level of filtering for adult content.
            filter (Optional[str]): Defines if the filters for 'Similar Results' and 'Omitted Results' are on or off.
            as_vis (Optional[str]): Defines whether to include citations or not.

        Returns:
            :class:`dict`: article information
                - title: a list of the titles of the three selected papers
                - cited_by: a list of the citation numbers of the three selected papers
                - organic_id: a list of the organic results' ids of the three selected papers
                - pub_info: publication information of selected papers
        """
        return super().search_google_scholar(query, cites, as_ylo, as_yhi,
                                             scisbd, cluster, hl, lr, start,
                                             num, as_sdt, safe, filter, as_vis)

    @tool_api(explode_return=True)
    @aioify
    def get_author_information(self,
                               author_id: str,
                               hl: Optional[str] = None,
                               view_op: Optional[str] = None,
                               sort: Optional[str] = None,
                               citation_id: Optional[str] = None,
                               start: Optional[int] = None,
                               num: Optional[int] = None,
                               no_cache: Optional[bool] = None,
                               async_req: Optional[bool] = None,
                               output: Optional[str] = None) -> dict:
        """Search for an author's information by author's id provided by get_author_id.

        Args:
            author_id (str): Required. The ID of an author.
            hl (Optional[str]): The language to use for the Google Scholar Author search. Default is 'en'.
            view_op (Optional[str]): Used for viewing specific parts of a page.
            sort (Optional[str]): Used for sorting and refining articles.
            citation_id (Optional[str]): Used for retrieving individual article citation.
            start (Optional[int]): Defines the result offset. Default is 0.
            num (Optional[int]): Defines the number of results to return. Default is 20.
            no_cache (Optional[bool]): Forces SerpApi to fetch the results even if a cached version is already present. Default is False.
            async_req (Optional[bool]): Defines the way you want to submit your search to SerpApi. Default is False.
            output (Optional[str]): Defines the final output you want. Default is 'json'.

        Returns:
            :class:`dict`: author information
                * name: author's name
                * affliation: the affliation of the author
                * articles: at most 3 articles by the author
                * website: the author's homepage url
        """
        return super().get_author_information(author_id, hl, view_op, sort,
                                              citation_id, start, num,
                                              no_cache, async_req, output)

    @tool_api(explode_return=True)
    @aioify
    def get_citation_format(self,
                            q: str,
                            no_cache: Optional[bool] = None,
                            async_: Optional[bool] = None,
                            output: Optional[str] = 'json') -> dict:
        """Function to get MLA citation format by an identification of organic_result's id provided by search_google_scholar.

        Args:
            q (str): ID of an individual Google Scholar organic search result.
            no_cache (Optional[bool]): If set to True, will force SerpApi to fetch the Google Scholar Cite results even if a cached version is already present. Defaults to None.
            async_ (Optional[bool]): If set to True, will submit search to SerpApi and retrieve results later. Defaults to None.
            output (Optional[str]): Final output format. Set to 'json' to get a structured JSON of the results, or 'html' to get the raw html retrieved. Defaults to 'json'.

        Returns:
            :class:`dict`: citation format
                * authors: the authors of the article
                * citation: the citation format of the article
        """
        return super().get_citation_format(q, no_cache, async_, output)

    @tool_api(explode_return=True)
    @aioify
    def get_author_id(self,
                      mauthors: str,
                      hl: Optional[str] = 'en',
                      after_author: Optional[str] = None,
                      before_author: Optional[str] = None,
                      no_cache: Optional[bool] = False,
                      _async: Optional[bool] = False,
                      output: Optional[str] = 'json') -> dict:
        """The getAuthorId function is used to get the author's id by his or her name.

        Args:
            mauthors (str): Defines the author you want to search for.
            hl (Optional[str]): Defines the language to use for the Google Scholar Profiles search. It's a two-letter language code. (e.g., 'en' for English, 'es' for Spanish, or 'fr' for French). Defaults to 'en'.
            after_author (Optional[str]): Defines the next page token. It is used for retrieving the next page results. The parameter has the precedence over before_author parameter. Defaults to None.
            before_author (Optional[str]): Defines the previous page token. It is used for retrieving the previous page results. Defaults to None.
            no_cache (Optional[bool]): Will force SerpApi to fetch the Google Scholar Profiles results even if a cached version is already present. Defaults to False.
            _async (Optional[bool]): Defines the way you want to submit your search to SerpApi. Defaults to False.
            output (Optional[str]): Defines the final output you want. It can be set to 'json' (default) to get a structured JSON of the results, or 'html' to get the raw html retrieved. Defaults to 'json'.

        Returns:
            :class:`dict`: author id
                * author_id: the author_id of the author
        """
        return super().get_author_id(mauthors, hl, after_author, before_author,
                                     no_cache, _async, output)

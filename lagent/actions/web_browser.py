import asyncio
import re
import time
from typing import Optional, Type

from bs4 import BeautifulSoup
from pyppeteer import launch

from lagent.actions.base_action import BaseAction, tool_api
from lagent.actions.parser import BaseParser, JsonParser
from lagent.schema import ActionReturn, ActionStatusCode


class WebBrowser(BaseAction):
    """Wrapper around the Web Browser Tool.
    """

    def __init__(self,
                 timeout=5,
                 description: Optional[dict] = None,
                 parser: Type[BaseParser] = JsonParser,
                 enable=True):
        super().__init__(description, parser, enable)

        self.timeout = timeout
        self.search_results = None

    @tool_api
    def search(self, query):
        """BING search API. When you want to get information from the Internet, you can use it.
        Args:
            query (str): the search content
        """
        tool_return = ActionReturn(type=self.name)
        try:
            search_results = asyncio.get_event_loop().run_until_complete(
                self.google_search(query))
            search_dict = dict()
            for idx, res in enumerate(search_results):
                search_dict[idx] = res
            self.search_results = search_dict
            tool_return.result = [dict(type='text', content=str(search_dict))]
        except Exception as e:
            tool_return.errmsg = str(e)
        return tool_return

    async def google_search(self, query):
        browser = await launch(
            headless=True, options={'args': ['--no-sandbox']})
        page = await browser.newPage()
        await page.goto(f'https://www.bing.com/search?q={query}')
        await page.waitForNavigation()
        time.sleep(1)
        # await page.waitForSelector('#li.b_algo')  # 等待搜索结果页面加载
        # 获取搜索结果
        results = await page.evaluate('''() => {
            results = []
            let items = document.querySelectorAll('li.b_algo');
            items.forEach((item) => {
                let title = item.querySelector('h2').innerText;
                let url = item.querySelector('a').href;
                let summaryElement = item.querySelector('.b_caption p');
                let summary = summaryElement ? summaryElement.innerText : '';
                results.push({
                    'title': title,
                    'url': url,
                    'summary': summary
                });
            });
            return results;
        }''')
        await browser.close()
        return results

    @tool_api
    def select(self, select_ids):
        """After calling BING search API, you can provide the ID of the search results to get more detailed information
        if you want further information. Only call it if the search response do not have enough information for user's question

        Args:
            select_ids (List[int]): list of index you want to open, select at least 3, at most 5 pages to open, the value with the format of "[1,3,5]".
        """
        tool_return = ActionReturn(type=self.name)
        if self.search_results is None:
            raise Exception
        assert isinstance(select_ids,
                          list), 'select_ids must be list, like [1, 2, 3]'
        selected_list = []
        for select_id in select_ids:
            assert isinstance(select_id,
                              int), 'the element in select_id must be integer'
            if select_id not in self.search_results:
                raise Exception('select_id must be in search results')
            search_item = self.search_results[select_id]
            web_content = self.open_url(search_item['url'])
            search_item['content'] = web_content
            selected_list.append(search_item)
        self.search_results = None
        tool_return.result = [dict(type='text', content=str(selected_list))]
        tool_return.state = ActionStatusCode.SUCCESS
        return tool_return

    @tool_api
    def open_url(self, url):
        """If you want to get the information from the url provided by the user, use this tool.

        Args:
            url (str): the url of the webpage
        """
        print('Start Browsing: %s' % url)
        try:
            html = asyncio.get_event_loop().run_until_complete(
                self.open_webpage(url))
            text = BeautifulSoup(html, 'html.parser').get_text()
            cleaned_text = re.sub(r'\n+', '\n', text)
        except Exception as e:
            print('Error:', str(e))
            cleaned_text = ''
        return cleaned_text

    async def open_webpage(self, url):
        browser = await launch(
            headless=True, options={'args': ['--no-sandbox']})
        page = await browser.newPage()
        await page.goto(url, timeout=5000)
        time.sleep(1)
        content = await page.content()
        await browser.close()
        return content


if __name__ == '__main__':
    a = WebBrowser()
    res = a.search('opencompass')
    print(res)

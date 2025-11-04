module.exports = {
  write: {
    platform: 'feishu',
    feishu: {
      type: 'wiki',
      wikiId: process.env.FEISHU_WIKI_ID,
      appId: process.env.FEISHU_APP_ID,
      appSecret: process.env.FEISHU_APP_SECRET,
      sort: true,
      catalog: false
    }
  },
  deploy: {
    platform: 'local',
    local: {
      outputDir: './content/posts',
      filename: 'title',
      format: 'markdown',
      catalog: true,
      frontMatter: {
        enable: true,
        include: ['title', 'date'],
        timeFormat: 'yyyy-MM-dd HH:mm:ss',
        template: `---
title: "{{ title }}"
date: "{{ date }}"
---`
      }
    }
  },
  image: {
    enable: true,
    platform: 'local',
    local: {
      outputDir: './static/images',
      prefixKey: '/blogs/images/'
    }
  }
}


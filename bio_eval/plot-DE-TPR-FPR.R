
library(readr)
library(ggplot2)
library(magrittr)



data1 <- read_csv('/aml_dataset/bio-eval-paper/outputs/DE_data_tpr_0_k5.csv')

data_summary <- data.frame()

for(m in unique(data1$model)){

  tmp <- dplyr::filter(data1, model == m)
  for(e in unique(tmp$epsilon)){

    tmp_ep <- dplyr::filter(tmp, epsilon == e)

    data_summary <- rbind(data_summary, data.frame(
      mean = mean(tmp_ep$correct),
      sd = sd(tmp_ep$correct),
      model = m,
      epsilon = e
    )
    )
  }
}

data_summary$epsilon <- factor(data_summary$epsilon, levels = c('5', '10', '20', '50', '100', 'non-priv'))
data_summary$seed <- 'k5'

data2 <- read_csv('/aml_dataset/bio-eval-paper/outputs/DE_data_fpr_0_k5.csv')

data_summary2 <- data.frame()

for(m in unique(data2$model)){

  tmp <- dplyr::filter(data2, model == m)
  for(e in unique(tmp$epsilon)){

    tmp_ep <- dplyr::filter(tmp, epsilon == e)

    data_summary2 <- rbind(data_summary2, data.frame(
      mean = mean(tmp_ep$correct),
      sd = sd(tmp_ep$correct),
      model = m,
      epsilon = e
    )
    )
  }
}

data_summary2$epsilon <- factor(data_summary2$epsilon, levels = c('5', '10', '20', '50', '100', 'non-priv'))
data_summary2$seed <- 'k5'

data_summary$model[data_summary$model == 'PGM'] = 'Private-PGM'
data_summary2$model[data_summary2$model == 'PGM'] = 'Private-PGM'


g <- ggplot()+
  geom_line(data=data_summary, aes(x = epsilon, y = mean, color = model, group = model), linewidth=2)+
  geom_rect(data=NULL,aes(xmin=0,xmax=5,ymin=-Inf,ymax=Inf),
            fill="#b3d4ff", color=NA, alpha=.1)+
  geom_line(data=data_summary, aes(x = epsilon, y = mean, color = model, group = model), linewidth=2)+
  geom_point(data=data_summary, aes(x = epsilon, y = mean, color = model, group = model),size=3)+
  
  geom_line(data=data_summary2, aes(x = epsilon, y = mean, color = model, group = model), linewidth=2, alpha=.1, linetype = "dashed")+
  geom_rect(data=NULL,aes(xmin=0,xmax=5,ymin=-Inf,ymax=Inf),
            fill="#b3d4ff", color=NA, alpha=.1)+
  geom_line(data=data_summary2, aes(x = epsilon, y = mean, color = model, group = model), linewidth=2, alpha=.7, linetype = "dashed")+
  geom_point(data=data_summary2, aes(x = epsilon, y = mean, color = model, group = model),size=3, alpha=.7)+
  ylim(c(0,1))+
  theme_bw()+
  scale_color_manual(breaks= c('Privsyn', 'VAE', 'RONGauss', 'GAN', 'Private-PGM'), values = c('#9467bd', '#1f77b4', '#2ca02c', '#ff7f0e', '#d62728'))+
  ggtitle('DE-Gene Preservation')+
  ylab('Mean DE-Gene Preservation')
g






data1 <- read_csv('/aml_dataset/bio-eval-paper/outputs/DE_data_tpr_0_k1000.csv')

data_summary <- data.frame()

for(m in unique(data1$model)){

  tmp <- dplyr::filter(data1, model == m)
  for(e in unique(tmp$epsilon)){

    tmp_ep <- dplyr::filter(tmp, epsilon == e)

    data_summary <- rbind(data_summary, data.frame(
      mean = mean(tmp_ep$correct),
      sd = sd(tmp_ep$correct),
      model = m,
      epsilon = e
    )
    )
  }
}

data_summary$epsilon <- factor(data_summary$epsilon, levels = c('5', '10', '20', '50', '100', 'non-priv'))
data_summary$seed <- 'k1000'


data2 <- read_csv('/aml_dataset/bio-eval-paper/outputs/DE_data_fpr_0_k1000.csv')

data_summary2 <- data.frame()

for(m in unique(data2$model)){

  tmp <- dplyr::filter(data2, model == m)
  for(e in unique(tmp$epsilon)){

    tmp_ep <- dplyr::filter(tmp, epsilon == e)

    data_summary2 <- rbind(data_summary2, data.frame(
      mean = mean(tmp_ep$correct),
      sd = sd(tmp_ep$correct),
      model = m,
      epsilon = e
    )
    )
  }
}

data_summary2$epsilon <- factor(data_summary2$epsilon, levels = c('5', '10', '20', '50', '100', 'non-priv'))
data_summary2$seed <- 'k5'

data_summary$model[data_summary$model == 'PGM'] = 'Private-PGM'
data_summary2$model[data_summary2$model == 'PGM'] = 'Private-PGM'



g2 <- ggplot()+
  geom_line(data=data_summary, aes(x = epsilon, y = mean, color = model, group = model), linewidth=2)+
  geom_rect(data=NULL,aes(xmin=0,xmax=5,ymin=-Inf,ymax=Inf),
            fill="#b3d4ff", color=NA, alpha=.1)+
  geom_line(data=data_summary, aes(x = epsilon, y = mean, color = model, group = model), linewidth=2)+
  geom_point(data=data_summary, aes(x = epsilon, y = mean, color = model, group = model),size=3)+
  
  geom_line(data=data_summary2, aes(x = epsilon, y = mean, color = model, group = model), linewidth=2, alpha=.1, linetype = "dashed")+
  geom_rect(data=NULL,aes(xmin=0,xmax=5,ymin=-Inf,ymax=Inf),
            fill="#b3d4ff", color=NA, alpha=.1)+
  geom_line(data=data_summary2, aes(x = epsilon, y = mean, color = model, group = model), linewidth=2, alpha=.7, linetype = "dashed")+
  geom_point(data=data_summary2, aes(x = epsilon, y = mean, color = model, group = model),size=3, alpha=.7)+
  
  ylim(c(0,1))+
  theme_bw()+
  scale_color_manual(breaks= c('Privsyn', 'VAE', 'RONGauss', 'GAN', 'Private-PGM'), values = c('#9467bd', '#1f77b4', '#2ca02c', '#ff7f0e', '#d62728'))+
  ggtitle('')+
  ylab('Mean DE-Gene Preservation')
g2






cp1 <- cowplot::plot_grid(g, g2, nrow=1)
cp1

png(file = paste0('~/Documents/pgg/DE-plots-TPR-FPR.png'), bg = "transparent",
    width = 30, height = 10, units = "cm", res=300)
plot(cp1)
dev.off()

